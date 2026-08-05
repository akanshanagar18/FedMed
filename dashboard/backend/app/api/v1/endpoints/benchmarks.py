"""
Module: dashboard.backend.app.api.v1.endpoints.benchmarks

Purpose:
REST API endpoints for managing multi-experiment benchmark suites and retrieving leaderboards.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.database.session import get_db
from app.services.benchmark import BenchmarkService
from common.schemas import Benchmark, SuccessResponse, ErrorResponse

router = APIRouter()


@router.get("", response_model=SuccessResponse)
async def list_benchmarks(db: Session = Depends(get_db)):
    """Retrieves all registered benchmarks."""
    benchmarks = BenchmarkService.list_benchmarks(db)
    return SuccessResponse(
        message="Benchmarks retrieved successfully",
        data=[b.model_dump(mode="json") for b in benchmarks],
    )


@router.get("/{benchmark_id}", response_model=SuccessResponse)
async def get_benchmark(benchmark_id: str, db: Session = Depends(get_db)):
    """Retrieves metadata for a specific benchmark ID."""
    bench = BenchmarkService.get_benchmark(db, benchmark_id)
    if not bench:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark '{benchmark_id}' not found",
        )
    return SuccessResponse(
        message="Benchmark retrieved successfully",
        data=bench.model_dump(mode="json"),
    )


@router.post("", response_model=SuccessResponse)
async def create_benchmark(bench: Benchmark, db: Session = Depends(get_db)):
    """Registers a new benchmark suite."""
    saved_model = BenchmarkService.create_benchmark(db, bench)
    saved_schema = BenchmarkService.get_benchmark(db, saved_model.benchmark_id)
    return SuccessResponse(
        message="Benchmark suite registered successfully",
        data=saved_schema.model_dump(mode="json") if saved_schema else bench.model_dump(mode="json"),
    )


@router.get("/{benchmark_id}/leaderboard", response_model=SuccessResponse)
async def get_benchmark_leaderboard(benchmark_id: str, db: Session = Depends(get_db)):
    """Computes and returns the leaderboard for a benchmark run."""
    leaderboard = BenchmarkService.generate_leaderboard(db, benchmark_id)
    return SuccessResponse(
        message=f"Leaderboard for benchmark '{benchmark_id}' computed successfully",
        data=[entry.model_dump(mode="json") for entry in leaderboard],
    )


@router.delete("/{benchmark_id}", response_model=SuccessResponse)
async def delete_benchmark(benchmark_id: str, db: Session = Depends(get_db)):
    """Deletes a benchmark suite and associated mappings."""
    deleted = BenchmarkService.delete_benchmark(db, benchmark_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark '{benchmark_id}' not found",
        )
    return SuccessResponse(
        message=f"Benchmark '{benchmark_id}' deleted successfully",
        data={"benchmark_id": benchmark_id},
    )
