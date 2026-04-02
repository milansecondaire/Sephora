import numpy as np
import pandas as pd
from src.preprocessing import _clean_primary_axe

def test_clean_primary_axe():
    assert _clean_primary_axe("SKINCARE") == "SKINCARE"
    assert _clean_primary_axe("SKINCARE|FRAGRANCE") == "SKINCARE"
    assert _clean_primary_axe("MAEK UP") == "MAKE UP"
    assert _clean_primary_axe("MAEK UP|SKINCARE") == "MAKE UP"
    assert _clean_primary_axe("RANDOM") == "OTHERS"
    assert pd.isna(_clean_primary_axe(np.nan))
