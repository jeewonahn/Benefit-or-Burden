@echo off
setlocal DisableDelayedExpansion
title TORS experiment
if "%~1"=="" (
    echo Usage: RUN_EXPERIMENT.cmd scripts\script_name.py [arguments]
    exit /b 2
)
if not defined TORS_PYTHON set "TORS_PYTHON=python"
pushd "%~dp0"
"%TORS_PYTHON%" -B -u %*
set "TORS_EXIT=%ERRORLEVEL%"
popd
exit /b %TORS_EXIT%
