#!/usr/bin/env python3
"""Асинхронный SSH-оркестратор инстансов (asyncssh).

Закрывает антипаттерн «timeout ssh … + sleep + blind tail»:
- exit code команды приходит СРАЗУ по её завершении (1 сек или час — неважно);
- вывод стримится в реальном времени — ошибка видна в момент возникновения;
- timeout на каждый exec: зависшая команда не держит нас часами;
- retry/reconnect: обрыв канала к дешёвому инстансу ловится как exception.

MCP-вызовы (rent/destroy) НЕ трогаются — оркестратор управляет только SSH/SFTP.

Использование:
  # probe: факт-конфиг + замер канала (exit 0/1/2 — как у speed-test.sh)
  python3 instance_orchestrator.py probe --host ssh1.qudata.ai --port 28958 [--min-mbps 5] [--timeout 240]

  # run: стриминг произвольной команды, exit code из инстанса
  python3 instance_orchestrator.py run --host … --port … -- bash /workspace/scripts/depth_pass_da3.py …

  # tail: живой стриминг лога (возвращает exit 0 по EOF, 124 по таймауту)
  python3 instance_orchestrator.py tail --host … --port … --path /workspace/logs/stereo_pass.log [--timeout 600]

  # watch: tail с детектом зависания (нет новых строк N сек → рестарт-подсказка)
  python3 instance_orchestrator.py watch --host … --port … --path … [--idle 120]
"""
import argparse
import asyncio
import os
import sys
import time

import asyncssh

DEFAULT_KEY = os.path.expanduser("~/.ssh/id_ed25519")
SPEED_TEST_REMOTE = "/workspace/scripts/speed-test.sh"


async def connect(host: str, port: int, user: str = "root",
                  key: str = DEFAULT_KEY, retries: int = 3,
                  retry_delay: float = 10.0) -> asyncssh.SSHClientConnection:
    """Коннект с ретраями: к дешёвым инстансам первый коннект часто в отказ."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return await asyncssh.connect(
                host, port=port, username=user,
                client_keys=[key], known_hosts=None,  # инстансы одноразовые
                connect_timeout=30,
            )
        except (asyncssh.DisconnectError, OSError) as e:
            last_err = e
            print(f"[connect] попытка {attempt}/{retries} не удалась: {e}",
                  file=sys.stderr)
            if attempt < retries:
                await asyncio.sleep(retry_delay)
    raise ConnectionError(f"Не удалось подключиться к {host}:{port}: {last_err}")


async def stream_process(proc: asyncssh.SSHClientProcess,
                         idle_timeout: float | None = None) -> int:
    """Стриминг stdout/stderr вживую; возвращает exit code.
    idle_timeout: если ни одной строки за N сек — warning (процесс завис?)."""
    async def pump(stream, tag: str) -> None:
        while not stream.at_eof():
            line = await stream.readline()
            if line:
                sys.stdout.write(f"{tag}{line}")
                sys.stdout.flush()
    tasks = [asyncio.create_task(pump(proc.stdout, ""))]
    if proc.stderr is not None:
        tasks.append(asyncio.create_task(pump(proc.stderr, "!! ")))
    last_activity = time.monotonic()
    while not (proc.stdout.at_eof() and
               (proc.stderr is None or proc.stderr.at_eof())):
        done, _ = await asyncio.wait(tasks, timeout=1.0, return_when=asyncio.FIRST_COMPLETED)
        if done:
            for t in done:
                if t.exception():
                    raise t.exception()
                tasks.remove(t)
        elif idle_timeout and time.monotonic() - last_activity > idle_timeout:
            print(f"\n[watch] нет вывода > {idle_timeout}s — процесс завис? "
                  f"(exit пока не получен)", file=sys.stderr)
            last_activity = time.monotonic()  # предупреждать раз в idle_timeout
    await proc.wait()
    for t in tasks:
        t.cancel()
    return proc.exit_status or 0


async def cmd_run(args: argparse.Namespace) -> int:
    conn = await connect(args.host, args.port, retries=args.retries)
    try:
        proc = await conn.create_process(args.command)
        return await asyncio.wait_for(
            stream_process(proc, idle_timeout=args.idle),
            timeout=args.timeout,
        )
    except asyncio.TimeoutError:
        print(f"[run] TIMEOUT {args.timeout}s — команда не завершилась", file=sys.stderr)
        return 124
    finally:
        conn.close()


async def cmd_probe(args: argparse.Namespace) -> int:
    """Пробный прогон: факт-конфиг + канал. Решение гасить/работать сразу."""
    conn = await connect(args.host, args.port, retries=args.retries)
    try:
        # 1) гарантируем наличие speed-test.sh (в полном образе уже есть;
        #    на голом инстансе — заливаем SFTP)
        async with conn.start_sftp() as sftp:
            try:
                await sftp.stat(SPEED_TEST_REMOTE)
            except asyncssh.SFTPNoSuchFile:
                local = os.path.join(os.path.dirname(__file__), "speed-test.sh")
                print(f"[probe] {SPEED_TEST_REMOTE} нет на инстансе, заливаю…")
                await sftp.put(local, SPEED_TEST_REMOTE)

        # 2) REAL-режим: факт-конфиг (быстро, exit по завершению)
        proc = await conn.create_process(f"REAL=1 bash {SPEED_TEST_REMOTE}")
        await stream_process(proc, idle_timeout=30)
        if proc.exit_status != 0:
            print(f"[probe] факт-конфиг: GPU отсутствует/сломан → гасить (exit {proc.exit_status})")
            return 2

        # 3) канальный замер (2-3 мин) — решение по каналу тут же
        proc = await conn.create_process(f"bash {SPEED_TEST_REMOTE} {args.min_mbps}")
        return await asyncio.wait_for(stream_process(proc, idle_timeout=45),
                                      timeout=args.timeout)
    except asyncio.TimeoutError:
        print(f"[probe] TIMEOUT {args.timeout}s — замер не завершился, гасить",
              file=sys.stderr)
        return 1
    finally:
        conn.close()


async def cmd_tail(args: argparse.Namespace) -> int:
    conn = await connect(args.host, args.port, retries=args.retries)
    try:
        cmd = f"tail -F -n {args.lines} {args.path}"
        if args.follow is False:
            cmd = f"tail -n {args.lines} {args.path}"
        proc = await conn.create_process(cmd)
        return await asyncio.wait_for(stream_process(proc, idle_timeout=None),
                                      timeout=args.timeout)
    except asyncio.TimeoutError:
        print(f"[tail] TIMEOUT {args.timeout}s — файл продолжает расти",
              file=sys.stderr)
        return 124
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--host", required=True)
        p.add_argument("--port", type=int, required=True)
        p.add_argument("--user", default="root")
        p.add_argument("--retries", type=int, default=3)

    p = sub.add_parser("probe", help="факт-конфиг + скорость канала")
    common(p)
    p.add_argument("--min-mbps", type=float, default=5.0)
    p.add_argument("--timeout", type=float, default=240.0)
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("run", help="запустить команду со стримингом")
    common(p)
    p.add_argument("command")
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("--idle", type=float, default=None,
                   help="предупреждать, если вывода нет N сек")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("tail", help="живой стриминг файла")
    common(p)
    p.add_argument("--path", required=True)
    p.add_argument("--lines", type=int, default=20)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--follow", type=bool, default=True)
    p.set_defaults(func=cmd_tail)

    args = ap.parse_args()
    try:
        code = asyncio.run(args.func(args))
    except (ConnectionError, asyncssh.Error, OSError) as e:
        print(f"[orchestrator] ошибка: {e}", file=sys.stderr)
        sys.exit(255)
    sys.exit(code or 0)


if __name__ == "__main__":
    main()
