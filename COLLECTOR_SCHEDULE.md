# Daily Collector Scheduling

The collector is append-only and idempotent. It stores raw responses under `data/raw/`
and logs each run in `data/collector_runs.jsonl`.

Safety rule for backtests and live decisions:

- A bot at decision time `T` may only use rows with `available_at_utc <= T - 1 hour`.
- News rows are forward-collected only and include both `published_at_utc` and `collected_at_utc`.

## Manual Daily Command

Run this from `C:\Users\saipr\Music\Ai Bot`:

```powershell
& 'C:\Users\saipr\AppData\Local\Programs\Python\Python314\python.exe' collect_daily_data.py --fear-greed --fear-greed-limit 7 --funding --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT --funding-limit 20 --news --news-max-records 25 --rss --report
```

## Windows Task Scheduler

1. Open Task Scheduler.
2. Choose "Create Basic Task".
3. Name it `Crypto Bot Daily Collector`.
4. Trigger: Daily.
5. Pick a time after your intended decision time. If the bot decides at 00:00 UTC,
   schedule collection at least one hour later.
6. Action: Start a program.
7. Program/script:

```text
C:\windows\System32\WindowsPowerShell\v1.0\powershell.exe
```

8. Add arguments:

```text
-NoProfile -ExecutionPolicy Bypass -Command "& 'C:\Users\saipr\AppData\Local\Programs\Python\Python314\python.exe' collect_daily_data.py --fear-greed --fear-greed-limit 7 --funding --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT --funding-limit 20 --news --news-max-records 25 --rss --report"
```

9. Start in:

```text
C:\Users\saipr\Music\Ai Bot
```

10. After saving, right-click the task and choose "Run" once. Check
    `data/collector_runs.jsonl` for failures.

