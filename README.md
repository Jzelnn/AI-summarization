Folder Structure:

summarization_ner_app/
│
├── backend/
│   ├── main.py
│   ├── input/
│   │   ├──test_stt.py
│   │   ├──speech_to_text.py
│   │   └──__pycache__/
│   ├── nlp/
│   │   ├──__init__.py
│   │   ├──fine_tune_ner.py
│   │   ├──ner.py
│   │   ├──pipeline.py
│   │   ├──summarizer.py
│   │   ├──test_nlp.py
│   │   ├──test_pipeline.py
│   │   └──__pycache__/
│   └── utils/
│       ├──__init__.py
│       ├──file_reader.py
│       ├──test_file_reader.py
│       └──__pycache__/
│
└── frontend/
    ├──index.html
    ├──script.js
    └──style.css
