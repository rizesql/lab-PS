import logging
import time
from pathlib import Path

import numpy as np
import PIL.Image
import typer

from jpeg import compressor, image, writer
from jpeg.image import RGB

cli = typer.Typer(add_completion=False)


@cli.command()
def img(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to input image"),
    output_path: Path = typer.Option(None, "--output", "-o", help="Output .jpg path (default: input + .jpg)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show processing details"),
):
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if output_path is None:
        output_path = input_path.with_suffix(".jpg")

    if verbose:
        typer.secho(f"Processing: {input_path}", fg=typer.colors.BLUE)

    try:
        raw = PIL.Image.open(input_path).convert("RGB")
        img: image.Image[RGB] = np.array(raw).view(image.Image)  # type: ignore

        if verbose:
            typer.echo(f"Image Dimensions: {img.shape[1]}x{img.shape[0]}")

        start_time = time.perf_counter()
        meta, data = compressor.compress(img)

        with open(output_path, "wb") as f:
            writer.to_file(f, meta, data)

        end_time = time.perf_counter()

        if verbose:
            elapsed = end_time - start_time
            original_size = input_path.stat().st_size
            compressed_size = output_path.stat().st_size
            ratio = (1 - (compressed_size / original_size)) * 100

            typer.secho(f"Success! Saved to {output_path}", fg=typer.colors.GREEN)
            typer.echo(f"Original Size:     {original_size / 1024:.2f} KB")
            typer.echo(f"Compressed Size:   {compressed_size / 1024:.2f} KB")
            typer.echo(f"Compression Ratio: {ratio:.1f}%")
            typer.echo(f"Time Elapsed:      {elapsed:.4f} s")

    except Exception as e:
        typer.secho(f"Error during compression: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@cli.command()
def mse():
    typer.echo("Done")
