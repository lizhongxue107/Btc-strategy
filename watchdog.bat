@echo off
chcp 65001 >nul
title BTC 自动交易 - 看门狗

echo ========================================
echo   BTC 自动交易 看门狗
echo   进程退出后自动重启
echo ========================================

:restart
echo [%date% %time%] 启动机器人...
set HTTPS_PROXY=http://127.0.0.1:7890
set HTTP_PROXY=http://127.0.0.1:7890
set PYTHONIOENCODING=utf-8

python -u -W ignore auto_trade_btc.py

echo [%date% %time%] 机器人已停止, 10秒后重启...
echo 按 Ctrl+C 退出看门狗
timeout /t 10 /nobreak >nul
goto restart
