@echo off
REM ============================================================================
REM  setup_nightly_venv.bat
REM
REM  Build an ISOLATED venv ("venv-nightly") next to OneTrainer's normal "venv",
REM  using a PyTorch *nightly* built against CUDA 12.9. The goal is to find out
REM  whether a newer torch/cuBLAS/cuDNN finally dispatches NATIVE Blackwell
REM  (sm_120) kernels instead of the Ampere (sm_80) fallback kernels our current
REM  stable 2.9.1+cu128 wheel uses for both attention and the projection GEMMs.
REM
REM  This does NOT touch the existing "venv" -- your normal OneTrainer install is
REM  left exactly as-is. Run this from the OneTrainer repo root:
REM
REM      scripts\setup_nightly_venv.bat
REM
REM  Then validate with the probe (the decisive readout is the "kernel identity"
REM  section -- look for sm120 vs sm80):
REM
REM      venv-nightly\Scripts\python.exe scripts\sdpa_backend_probe.py
REM
REM  Requires the Python launcher with a 3.13 interpreter (py -3.13). Adjust the
REM  PYVER below if you use a different minor version.
REM ============================================================================

setlocal
set VENV=venv-nightly
set PYVER=-3.13
set NIGHTLY_INDEX=https://download.pytorch.org/whl/nightly/cu129

cd /d "%~dp0\.."
echo == OneTrainer root: %CD%

if exist "%VENV%\Scripts\python.exe" (
    echo == %VENV% already exists -- reusing it. Delete the folder to start clean.
) else (
    echo == Creating %VENV% with py %PYVER% ...
    py %PYVER% -m venv "%VENV%"
    if errorlevel 1 (
        echo !! Failed to create venv. Is "py %PYVER%" installed?  ^(try: py -0p^)
        exit /b 1
    )
)

set PY=%VENV%\Scripts\python.exe

echo == Upgrading pip / wheel ...
"%PY%" -m pip install --upgrade pip wheel
if errorlevel 1 exit /b 1

REM --- 1) nightly torch FIRST so nothing downstream can downgrade it ----------
echo == Installing nightly torch + torchvision from cu129 ...
"%PY%" -m pip install --pre torch torchvision --index-url %NIGHTLY_INDEX%
if errorlevel 1 (
    echo !! Nightly torch install failed -- check the index URL / network.
    exit /b 1
)

REM --- 2) OneTrainer's global deps (includes "-e ../mgds", no torch pin) -------
echo == Installing requirements-global.txt (incl. editable mgds fork) ...
"%PY%" -m pip install -r requirements-global.txt
if errorlevel 1 exit /b 1

REM --- 3) CUDA-side deps EXCEPT the torch/torchvision pins (filter them out) ---
REM     Keeps onnxruntime-gpu, triton-windows, bitsandbytes; drops the two pins
REM     that would otherwise clobber our nightly torch back to 2.9.1+cu128.
set FILTERED=%TEMP%\ot_req_cuda_notorch.txt
findstr /v /i /r /c:"^torch==" /c:"^torchvision==" requirements-cuda.txt > "%FILTERED%"
echo == Installing requirements-cuda.txt (torch/torchvision pins removed) ...
"%PY%" -m pip install -r "%FILTERED%"
if errorlevel 1 exit /b 1
del "%FILTERED%" 2>nul

REM --- 4) belt-and-suspenders: make sure OUR mgds fork is the editable one -----
echo == (Re)installing editable mgds fork to preserve local changes ...
"%PY%" -m pip install -e ..\mgds
if errorlevel 1 exit /b 1

REM --- 5) verification: what did we actually get? -----------------------------
echo.
echo ================= verification =================
"%PY%" -c "import torch, torchvision; print('torch       :', torch.__version__); print('torchvision :', torchvision.__version__); print('cuda        :', torch.version.cuda); print('cudnn       :', torch.backends.cudnn.version()); print('arch_list   :', torch.cuda.get_arch_list()); print('device      :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA VISIBLE')"
"%PY%" -c "import mgds, mgds.perf_probe; print('mgds        :', mgds.__file__)"
echo ================================================
echo.
echo == Done. Next, run the probe and check the 'kernel identity' section:
echo.
echo       %VENV%\Scripts\python.exe scripts\sdpa_backend_probe.py
echo.
echo    sm120 in the cuDNN attn / bf16 GEMM rows  =^> native Blackwell kernels (upgrade worth it)
echo    still sm80                                =^> no win, stay on the stable venv
echo.
echo    NOTE: the triton-windows pin is kept from requirements-cuda.txt. If a full
echo    OneTrainer run later errors inside torch.compile on this nightly, that pin
echo    is the likely culprit -- the probe itself does NOT compile, so it is
echo    unaffected and remains a valid kernel-arch test regardless.

endlocal
