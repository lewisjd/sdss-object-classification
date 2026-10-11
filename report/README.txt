SDSS report for Overleaf

1. In Overleaf, create a new project by uploading SDSS_report_overleaf.zip.
2. Use main.tex as the main document and pdfLaTeX as the compiler.
3. Add your name inside \author{} near the top of main.tex.
4. Compile the project. References are included in main.tex; no BibTeX step
   or additional files are required.

The ZIP contains main.tex, this README and three figures.
The compiled preview is supplied separately as SDSS_report.pdf.

The results use all 50,000 objects in data/sdss_sample.csv. To recreate them,
run the numbered scripts in src/ as described in the project README.
Tables are saved in outputs/tables/ and plots in outputs/figures/.
The report's results are unchanged by the code simplification.

This is a saved write-up. If you change the analysis later, update main.tex
and the three matching images in report/figures/ before rebuilding the ZIP.
Only the main results are included in the report; the other project outputs
provide additional diagnostics.
