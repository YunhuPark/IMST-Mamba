@echo off
REM ============================================================
REM  Compile IMST-Mamba paper to PDF
REM  Run from the Triage\ directory: compile_paper.bat
REM ============================================================

echo [1/4] Generating figures...
python scripts/generate_figures.py
if %ERRORLEVEL% NEQ 0 (
    echo [warn] Figure generation failed - PDF will have missing figures
)

echo [2/4] Running pdflatex (pass 1)...
pdflatex -interaction=nonstopmode paper_imst_mamba_standalone.tex
if %ERRORLEVEL% NEQ 0 (
    echo [error] pdflatex failed. Check paper_imst_mamba_standalone.log
    pause
    exit /b 1
)

echo [3/4] Running bibtex...
bibtex paper_imst_mamba_standalone
if %ERRORLEVEL% NEQ 0 (
    echo [warn] bibtex had issues - references may be missing
)

echo [4/4] Running pdflatex (passes 2 and 3 for cross-references)...
pdflatex -interaction=nonstopmode paper_imst_mamba_standalone.tex
pdflatex -interaction=nonstopmode paper_imst_mamba_standalone.tex

echo.
echo ============================================================
if exist paper_imst_mamba_standalone.pdf (
    echo  SUCCESS: paper_imst_mamba_standalone.pdf created
    echo  Open: start paper_imst_mamba_standalone.pdf
) else (
    echo  FAILED: PDF not created. Check .log file for errors.
)
echo ============================================================
pause
