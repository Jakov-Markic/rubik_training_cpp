@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set VENV=c:\Users\User\Desktop\SolveMeCube\rubik_training\venv\Scripts
"%VENV%\cmake.exe" -B build -G Ninja -DCMAKE_MAKE_PROGRAM="%VENV%\ninja.exe" -DCMAKE_BUILD_TYPE=Release %*
