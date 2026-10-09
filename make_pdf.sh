#!/bin/zsh
# usage: make_pdf.sh  -> downloads the HTML report written by 91_report, prints it to PDF with headless Chrome (print CSS:
# A4 landscape, one slide per page), and uploads both formats to the volume and the workspace report/ folder.
set -e
P=${RETRIEVAL_PROFILE:-DEFAULT}; VOL=dbfs:/Volumes/retail_consumer_goods/product_search/raw/reports
WS=/Users/$(databricks -p $P current-user me -o json | python3 -c "import sys, json; print(json.load(sys.stdin)['userName'])")/ai-search-product-retrieval/report
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"; DIR=${0:A:h}/report; mkdir -p $DIR
databricks -p $P fs cp --overwrite $VOL/product_retrieval_report.html $DIR/product_retrieval_report.html
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf=$DIR/product_retrieval_report.pdf file://$DIR/product_retrieval_report.html
databricks -p $P fs cp --overwrite $DIR/product_retrieval_report.pdf $VOL/product_retrieval_report.pdf
for f in html pdf; do databricks -p $P workspace import $WS/product_retrieval_report.$f --file $DIR/product_retrieval_report.$f --format AUTO --overwrite; done
echo "$DIR/product_retrieval_report.pdf"
