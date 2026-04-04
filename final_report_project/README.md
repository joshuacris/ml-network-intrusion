# Final Report Project

This is the working LaTeX report folder for the team.

Work here, not in `RGNIYD_Template_AI_ML_/`.

## Where To Edit

- Main entrypoint: `main.tex`
- Cover page: `Title/coverpage.tex`
- Abstract: `Title/Abstract.tex`
- Introduction: `Ch1-Introduction/Ch1-Introduction.tex`
- Data section: `Ch2-Data/Ch2-Data.tex`
- Model comparison: `Ch3-Model-Comparison/Ch3-Model-Comparison.tex`
- Originality and future work: `Ch4-Originality-Future-Work/Ch4-Originality-Future-Work.tex`
- Conclusion: `Ch5-Conclusion/Ch5-Conclusion.tex`
- References: `Ref/References.bib`
- Figures and plots: `Figures/`

## What You Need Installed

Minimum:

- A LaTeX distribution:
  - macOS: `TinyTeX` or `MacTeX`
  - Linux: `TeX Live`
  - Windows: `MiKTeX` or `TeX Live`
- `latexmk`

Recommended for editing:

- VS Code
- VS Code extension: `LaTeX Workshop`

Packages used by this report include:

- `fancyhdr`
- `caption`
- `setspace`
- `tocbibind`
- `natbib`
- `booktabs`
- `tabularx`

If your LaTeX distribution is reasonably complete, these should already be available. If not, install the missing packages through your TeX package manager.

## Build Locally

From the repo root:

```bash
cd "final_report_project"
latexmk -pdf -outdir=build main.tex
```

If you want a full clean rebuild:

```bash
cd "final_report_project"
latexmk -C
latexmk -pdf -outdir=build main.tex
```

## View The PDF

The compiled PDF is:

```bash
final_report_project/build/main.pdf
```

On macOS, you can open it with:

```bash
open build/main.pdf
```

## VS Code Workflow

1. Open the `final_report_project` folder in VS Code.
2. Open `main.tex`.
3. Run `LaTeX Workshop: Build LaTeX project`.
4. Run `LaTeX Workshop: View LaTeX PDF`.

## Notes

- Build from `main.tex`, not from an individual chapter file.
- The output directory is `build/`.
- If citations are changed, rerun `latexmk -pdf -outdir=build main.tex`; `latexmk` will handle BibTeX automatically.
- `RGNIYD_Template_AI_ML_/` is only the reference template folder and is gitignored.
