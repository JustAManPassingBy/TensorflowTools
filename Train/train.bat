@echo off

python tf_getresult.py
for /f "delims=" %%i in (result.txt) do echo %%i

set /p str=�����Ϸ��� �ƹ� Ű�� �Է��ϼ���