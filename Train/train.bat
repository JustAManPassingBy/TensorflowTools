@echo off

python tf_getresult.py
for /f "delims=" %%i in (result.txt) do echo %%i

set /p str=종료하려면 아무 키나 입력하세요