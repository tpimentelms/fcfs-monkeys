import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LANG_NAMES = {
    'simple': 'Simple English',
    'en': 'English',
    'fi': 'Finnish',
    'pt': 'Portuguese',
    'he': 'Hebrew',
    'id': 'Indonesian',
    'ta': 'Tamil',
    'tr': 'Turkish',
    'yo': 'Yoruba',
}
LANGUAGES = ['en', 'fi', 'he', 'id', 'pt', 'tr']

LEGEND_ORDER = {
    'Natural': 0,
    'FCFS': 2,
    'PolyFCFS': 4,
    'Caplan': 3,
    'IID': 3,
    'PolyCaplan': 5,
    'PolyIID': 5,
    ' ': 1,
}
