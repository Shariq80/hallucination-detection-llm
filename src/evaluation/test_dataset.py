test_data = [

    # =========================
    # SUPPORTED (TRUE CLAIMS)
    # =========================
    {"claim": "The Eiffel Tower is in Paris.", "label": "SUPPORTED"},
    {"claim": "Barack Obama was the 44th president of the United States.", "label": "SUPPORTED"},
    {"claim": "Water boils at 100 degrees Celsius at sea level.", "label": "SUPPORTED"},
    {"claim": "The Great Wall of China is visible from space is a myth.", "label": "SUPPORTED"},
    {"claim": "Python is a programming language.", "label": "SUPPORTED"},
    {"claim": "The capital of Japan is Tokyo.", "label": "SUPPORTED"},
    {"claim": "The human heart pumps blood.", "label": "SUPPORTED"},

    # =========================
    # REFUTED (FALSE CLAIMS)
    # =========================
    {"claim": "The Eiffel Tower is in Berlin.", "label": "REFUTED"},
    {"claim": "Barack Obama was born in Canada.", "label": "REFUTED"},
    {"claim": "Water boils at 50 degrees Celsius.", "label": "REFUTED"},
    {"claim": "The Sun revolves around the Earth.", "label": "REFUTED"},
    {"claim": "Python is only a snake species.", "label": "REFUTED"},
    {"claim": "The capital of France is Rome.", "label": "REFUTED"},
    {"claim": "Humans can breathe in space without assistance.", "label": "REFUTED"},

    # =========================
    # HARD CASES (CHALLENGING)
    # =========================
    {"claim": "The Eiffel Tower is one of the most visited monuments in the world.", "label": "SUPPORTED"},
    {"claim": "Albert Einstein failed mathematics in school.", "label": "REFUTED"},
    {"claim": "Shakespeare wrote many famous plays.", "label": "SUPPORTED"},
    {"claim": "Mount Everest is the tallest mountain on Earth above sea level.", "label": "SUPPORTED"},
    {"claim": "The Amazon rainforest produces most of the world's oxygen.", "label": "REFUTED"},
    {"claim": "Lightning never strikes the same place twice.", "label": "REFUTED"},
]