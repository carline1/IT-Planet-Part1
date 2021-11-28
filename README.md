# IT-Planet-Part1

The first part of the International Olympiad IT Planet, OCR

At this stage of the competition, it is necessary to develop a model that could extract certain features from the scanned document.

Main.py has a function extract_doc_features, which takes a document file as input and returns the following data:
result = {
  'red_areas_count': int, # number of red areas (stamps, print, etc.) on a scan
  'blue_areas_count': int, # number of blue areas (signatures, stamps, stamps) on the scan
  'text_main_title': str, # text of the main title of the page or ""
  'text_block': str, # text block of the page paragraph, only the first 10 words, or ""
  'table_cells_count': int, # unique number of cells (sum the number of cells in one or more tables)
}
