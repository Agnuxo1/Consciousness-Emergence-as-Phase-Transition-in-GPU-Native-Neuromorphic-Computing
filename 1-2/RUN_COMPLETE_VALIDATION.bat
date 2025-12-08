@echo off
REM ============================================================================
REM COMPLETE VALIDATION PIPELINE FOR EXPERIMENT 1
REM ============================================================================
REM
REM This script runs the entire validation workflow:
REM 1. Tests (30 min)
REM 2. Benchmarks (2 hours)
REM 3. Audits (1.5 hours)
REM 4. Report generation (1 min)
REM
REM Total time: ~4 hours
REM ============================================================================

echo ================================================================================
echo EXPERIMENT 1 - COMPLETE VALIDATION PIPELINE
echo ================================================================================
echo.
echo This will run:
echo   1. Unit and integration tests (~30 min)
echo   2. Accuracy benchmarks (~2 hours)
echo   3. Dual independent audit (~1.5 hours)
echo   4. Final report generation (~1 min)
echo.
echo Total estimated time: ~4 hours
echo.
echo ================================================================================
echo.

pause

cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo STEP 0: INSTALLING DEPENDENCIES
echo ================================================================================
echo.

python -m pip install -r requirements_testing.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo STEP 1: RUNNING TESTS
echo ================================================================================
echo.

python run_all_tests.py --verbose
if errorlevel 1 (
    echo WARNING: Some tests failed
    echo Continuing with benchmarks...
)

echo.
echo ================================================================================
echo STEP 2: RUNNING BENCHMARKS
echo ================================================================================
echo.

python run_all_benchmarks.py
if errorlevel 1 (
    echo WARNING: Some benchmarks failed
    echo Continuing with audits...
)

echo.
echo ================================================================================
echo STEP 3: RUNNING DUAL AUDIT
echo ================================================================================
echo.

python run_dual_audit.py
if errorlevel 1 (
    echo WARNING: Audit had issues
    echo Continuing with report generation...
)

echo.
echo ================================================================================
echo STEP 4: GENERATING FINAL REPORT
echo ================================================================================
echo.

python generate_final_report.py
if errorlevel 1 (
    echo ERROR: Failed to generate report
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo VALIDATION COMPLETE
echo ================================================================================
echo.
echo Results available in:
echo   - EXPERIMENT1_COMPREHENSIVE_REPORT.md
echo   - benchmark_accuracy_results.json
echo   - audit_energy_discrepancy.json
echo.
echo ================================================================================
echo.

pause
