import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import ffmpeg
import numpy as np
import typer
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from jpeg import compressor, image, writer
from jpeg.image import RGB

cli = typer.Typer(add_completion=False)


def setup_logging(verbose: bool):
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def format_size(size_bytes: int) -> str:
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def print_stats(input_path: Path, output_path: Path, duration: float, verbose: bool, quality: Optional[int] = None):
    original_size = input_path.stat().st_size
    compressed_size = output_path.stat().st_size
    ratio = (1 - (compressed_size / original_size)) * 100

    if verbose:
        typer.secho(f"Success! Saved to {output_path}", fg=typer.colors.GREEN)
        typer.echo(f"Original Size:      {format_size(original_size)}")
        typer.echo(f"Compressed Size:    {format_size(compressed_size)}")
        typer.echo(f"Compression Ratio:  {ratio:.1f}%")
        typer.echo(f"Time Elapsed:       {duration:.4f} s")
        if quality:
            typer.echo(f"Quality Used:       {quality}")
    else:
        typer.secho(f"Saved to {output_path}", fg=typer.colors.GREEN, nl=False)
        details = f"{ratio:.1f}% reduction"
        if quality:
            details = f"Q:{quality} | " + details
        typer.echo(f" ({details})")


def spinner_progress():
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    )


def load_img(path: Path) -> image.Image[RGB]:
    raw = cv2.imread(str(path))
    if raw is None:
        raise ValueError(f"Failed to load image: {path}")

    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    return np.array(raw).view(image.Image)  # type: ignore


@cli.command()
def img(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to input image"),
    output_path: Path = typer.Option(None, "--output", "-o", help="Output .jpg path (default: input + .jpg)"),
    quality: int = typer.Option(50, "--quality", "-q", help="JPEG quality (default: 50)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show processing details"),
):
    setup_logging(verbose)

    if output_path is None:
        output_path = input_path.with_suffix(".jpg")

    if verbose:
        typer.secho(f"Processing: {input_path}", fg=typer.colors.BLUE)

    try:
        with spinner_progress() as progress:
            task = progress.add_task(description="Loading image...", total=None)

            img = load_img(input_path)

            if verbose:
                progress.console.print(f"Image Dimensions: {img.shape[1]}x{img.shape[0]}")

            progress.update(task, description="Compressing...")
            start_time = time.perf_counter()

            meta, data = compressor.compress(img, quality)

            progress.update(task, description="Saving file...")
            with open(output_path, "wb") as f:
                writer.to_file(f, meta, data)

            elapsed = time.perf_counter() - start_time
            print_stats(input_path, output_path, elapsed, verbose, quality)

    except Exception as e:
        typer.secho(f"Error during compression: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@cli.command()
def mse(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to input image"),
    target_mse: float = typer.Argument(..., help="Target MSE threshold"),
    output_path: Path = typer.Option(None, "--output", "-o", help="Output .jpg path (default: input + .jpg)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show processing details"),
):
    setup_logging(verbose)

    if output_path is None:
        output_path = input_path.with_suffix(".jpg")

    if verbose:
        typer.secho(f"Processing: {input_path}", fg=typer.colors.BLUE)

    try:
        with spinner_progress() as progress:
            task = progress.add_task(description="Loading image...", total=None)

            img = load_img(input_path)

            if verbose:
                progress.console.print(f"Image Dimensions: {img.shape[1]}x{img.shape[0]}")

            progress.update(task, description=f"Finding optimal quality (Target MSE: {target_mse})...")

            start_opt = time.perf_counter()
            quality = compressor.find_opt_quality(img, target_mse)
            end_opt = time.perf_counter()

            if verbose:
                progress.console.print(
                    f"Optimal Quality Found: {quality} (took {end_opt - start_opt:.2f}s)", style="green"
                )

            progress.update(task, description=f"Compressing with Quality {quality}...")
            start_time = time.perf_counter()
            meta, data = compressor.compress(img, quality=quality)

            progress.update(task, description="Saving file...")
            with open(output_path, "wb") as f:
                writer.to_file(f, meta, data)

            elapsed = time.perf_counter() - start_time
            print_stats(input_path, output_path, elapsed, verbose, quality)

    except Exception as e:
        typer.secho(f"Error during compression: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@cli.command()
def vid(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to input image"),
    output_path: Path = typer.Option(None, "--output", "-o", help="Output .avi path (default: input + .avi)"),
    quality: int = typer.Option(50, "--quality", "-q", help="JPEG quality (default: 50)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show processing details"),
):
    setup_logging(verbose)

    if output_path is None:
        output_path = input_path.with_suffix(".avi")

    if verbose:
        typer.secho(f"Processing: {input_path}", fg=typer.colors.BLUE)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        typer.secho("Failed to open video file", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if verbose:
        typer.echo(f"Video Info: {W}x{H} @ {fps:.2f} fps, {total_frames} frames")

    frame_count = 0
    start_time = time.perf_counter()

    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ]
    if total_frames > 0:
        progress_columns.append(TimeRemainingColumn())

    process = (
        ffmpeg.input("pipe:0", format="mjpeg", framerate=fps)
        .output(
            str(output_path),
            vcodec="libx264",
            pix_fmt="yuv420p",
            profile="high",
            level="4.0",
            preset="medium",
            crf=18,
            movflags="+faststart",
        )
        .overwrite_output()
        .run_async(pipe_stdin=True, pipe_stderr=not verbose)
    )
    try:
        with Progress(*progress_columns, transient=True) as progress:
            task = progress.add_task("Compressing...", total=total_frames if total_frames > 0 else None)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                meta, frame = compressor.compress(frame, quality)  # type: ignore
                writer.to_file(process.stdin, meta, data=frame)

                progress.update(task, advance=1)
                frame_count += 1

    finally:
        if process.stdin:
            process.stdin.close()
        process.wait()

        cap.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_time

    print_stats(input_path, output_path, elapsed, verbose, quality)

    if verbose:
        if elapsed > 0:
            typer.echo(f"Average FPS: {frame_count / elapsed:.2f}")
