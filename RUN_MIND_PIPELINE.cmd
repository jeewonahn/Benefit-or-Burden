@echo off
setlocal DisableDelayedExpansion
if not defined TORS_PYTHON set "TORS_PYTHON=python"
"%TORS_PYTHON%" -B -u "%~dp0scripts\run_mind_pipeline.py" %*
exit /b %errorlevel%
