BardoThodol/
│
├── core/
│   ├── states.py              # Consolidación de bardo_states + post_mortem_states + utils/states
│   └── circuit_factory.py     # Consolidación de quantum_circuit + simulation + bardo_simulation
│
├── utils/
│   └── visualizations.py      # Visualizaciones movidas aquí
│
├── data/                      # Si usas datasets o parámetros
│
├── docs/
│   └── technical_guide.md     # Documentación técnica ampliada
│
├── tests/
│   ├── test_states.py
│   ├── test_circuit_factory.py
│   └── ...
│
├── BardoQuantum.tex           # Paper principal
├── references.bib             # Bibliografía
├── requirements.txt           # Versiones fijas
├── .gitignore                 # Agregar .env aquí
├── .env                       # Variables de entorno (NO subir a GitHub)
└── README.md
