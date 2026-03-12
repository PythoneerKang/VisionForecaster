import parameters as p

import pickle
import numpy as np
import torch


def extract_distance_matrix():
    """
    Load the precomputed correlation tensor and convert it into a
    z-scored distance matrix, keeping memory usage under control.

    Expected pkl shape : (T, 463, 463)  — 463 stocks, 6 will be removed.
    Output shape       : (T, 457, 457)  — matches model input size.
    """
    with open(
        "../Quasi_Differentiation_High_Temporal_Resolution_Cross_Correlations/Codes/"
        "Extract distance matrix (2017-2022) from pkl file/IQDw{}.pkl".format(p.w),
        "rb",
    ) as pkl_file:
        data1 = pickle.load(pkl_file).astype(np.float32)

    data1 = np.clip(data1, -1.0, 1.0)

    # Convert correlation → distance:  d = sqrt(2 * (1 − corr))
    distance_matrix = (2 * (1 - data1)) ** np.float32(0.5)

    del data1

    assert distance_matrix.shape[1] == 457 + 6, (
        f"Unexpected stock count {distance_matrix.shape[1]} in pkl; "
        f"expected 463 (457 stocks + 6 to be removed)."
    )

    # Removed tickers: ABMD (idx 6), CTVA (111), DOW (128), FOX (169),
    #                  FOXA (170), IR (225)
    bad_indices = [6, 111, 128, 169, 170, 225]
    distance_matrix = np.delete(distance_matrix, bad_indices, axis=1)
    distance_matrix = np.delete(distance_matrix, bad_indices, axis=2)

    assert distance_matrix.shape[1] == 457 and distance_matrix.shape[2] == 457, (
        f"Distance matrix stock dimensions {distance_matrix.shape[1:]} != (457, 457) "
        f"after removing bad indices."
    )

    # Global z-score standardisation
    mean = np.mean(distance_matrix, dtype=np.float32)
    std  = np.std(distance_matrix,  dtype=np.float32)
    distance_matrix = (distance_matrix - mean) / std

    return distance_matrix


# ─────────────────────────────────────────────────────────────────────────────
# GICS sector label file
# ─────────────────────────────────────────────────────────────────────────────

SP500_TICKERS_457 = [
    "A", "AAL", "AAP", "AAPL", "ABBV", "ABC", "ABT", "ACN", "ADBE", "ADI",
    "ADM", "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIV", "AIZ",
    "AJG", "AKAM", "ALB", "ALGN", "ALK", "ALL", "ALLE", "AMAT", "AMCR", "AMD",
    "AME", "AMGN", "AMP", "AMT", "AMZN", "ANET", "ANSS", "AON", "AOS", "APA",
    "APD", "APH", "APTV", "ARE", "ATO", "ATVI", "AVB", "AVGO", "AVY", "AWK",
    "AXP", "AZO", "BA", "BAC", "BAX", "BBY", "BDX", "BEN", "BIIB", "BK",
    "BKNG", "BKR", "BLK", "BMY", "BR", "BSX", "BWA", "BXP", "C", "CAG",
    "CAH", "CAT", "CB", "CBOE", "CBRE", "CCI", "CCL", "CDNS", "CDW", "CE",
    "CF", "CFG", "CHD", "CHRW", "CHTR", "CI", "CINF", "CL", "CLX", "CMA",
    "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC", "CNP", "COF", "COO", "COP",
    "COST", "COTY", "CPB", "CPRI", "CPRT", "CRM", "CSCO", "CSX", "CTAS",
    "CTSH", "CVS", "CVX", "D", "DAL", "DD", "DE", "DFS", "DG", "DGX",
    "DHI", "DHR", "DIS", "DISH", "DLR", "DLTR", "DOV", "DRI", "DTE", "DUK",
    "DVA", "DVN", "DXC", "EA", "EBAY", "ECL", "ED", "EFX", "EIX", "EL",
    "EMN", "EMR", "EOG", "EQIX", "EQR", "ES", "ESS", "ETN", "ETR", "EVRG",
    "EW", "EXC", "EXPD", "EXPE", "EXR", "F", "FANG", "FAST", "FCX", "FDX",
    "FE", "FFIV", "FIS", "FITB", "FLS", "FLT", "FMC", "FRT", "FTI", "FTNT",
    "FTV", "GD", "GE", "GILD", "GIS", "GL", "GLW", "GM", "GOOG", "GOOGL",
    "GPC", "GPN", "GPS", "GRMN", "GS", "GWW", "HAL", "HAS", "HBAN", "HBI",
    "HCA", "HD", "HES", "HIG", "HII", "HLT", "HOG", "HOLX", "HON", "HP",
    "HPE", "HPQ", "HRB", "HRL", "HSIC", "HST", "HSY", "HUM", "IBM", "ICE",
    "IDXX", "IEX", "IFF", "ILMN", "INCY", "INTC", "INTU", "IP", "IPG",
    "IPGP", "IQV", "IRM", "ISRG", "IT", "ITW", "IVZ", "J", "JBHT", "JCI",
    "JKHY", "JNJ", "JNPR", "JPM", "JWN", "K", "KEY", "KEYS", "KHC", "KIM",
    "KLAC", "KMB", "KMI", "KMX", "KO", "KR", "KSS", "L", "LDOS", "LEG",
    "LEN", "LH", "LHX", "LIN", "LKQ", "LLY", "LMT", "LNC", "LNT", "LOW",
    "LRCX", "LUV", "LVS", "LW", "LYB", "LYV", "M", "MA", "MAA", "MAR",
    "MAS", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MET", "MGM", "MHK",
    "MKC", "MKTX", "MLM", "MMC", "MMM", "MNST", "MO", "MOS", "MPC", "MRK",
    "MRO", "MS", "MSCI", "MSFT", "MSI", "MTB", "MTD", "MU", "NCLH", "NDAQ",
    "NEE", "NEM", "NFLX", "NI", "NKE", "NOC", "NOV", "NOW", "NRG", "NSC",
    "NTAP", "NTRS", "NUE", "NVDA", "NVR", "NWL", "NWS", "NWSA", "O", "ODFL",
    "OKE", "OMC", "ORCL", "ORLY", "OXY", "PAYC", "PAYX", "PCAR", "PEAK",
    "PEG", "PEP", "PFE", "PFG", "PG", "PGR", "PH", "PHM", "PKG", "PKI",
    "PLD", "PM", "PNC", "PNR", "PNW", "PPG", "PPL", "PRGO", "PRU", "PSA",
    "PSX", "PVH", "PWR", "PXD", "PYPL", "QCOM", "QRVO", "RCL", "RE", "REG",
    "REGN", "RF", "RHI", "RJF", "RL", "RMD", "ROK", "ROL", "ROP", "ROST",
    "RSG", "SBAC", "SBUX", "SCHW", "SEE", "SHW", "SJM", "SLB", "SLG", "SNA",
    "SNPS", "SO", "SPG", "SPGI", "SRE", "STE", "STT", "STX", "STZ", "SWK",
    "SWKS", "SYF", "SYK", "SYY", "T", "TAP", "TDG", "TEL", "TFC", "TFX",
    "TGT", "TJX", "TMO", "TMUS", "TPR", "TROW", "TRV", "TSCO", "TSN", "TTWO",
    "TXN", "TXT", "UA", "UAA", "UAL", "UDR", "UHS", "ULTA", "UNH", "UNM",
    "UNP", "UPS", "URI", "USB", "V", "VFC", "VLO", "VMC", "VNO", "VRSK",
    "VRSN", "VRTX", "VTR", "VZ", "WAB", "WAT", "WBA", "WDC", "WEC", "WELL",
    "WFC", "WHR", "WM", "WMB", "WMT", "WRB", "WRK", "WU", "WY", "WYNN",
    "XEL", "XOM", "XRAY", "XRX", "XYL", "YUM", "ZBH", "ZBRA", "ZION", "ZTS",
]

# ─────────────────────────────────────────────────────────────────────────────
# GICS sector → ticker mapping
# ─────────────────────────────────────────────────────────────────────────────
#
# Sources
# -------
# Primary   : Wikipedia, "List of S&P 500 companies"
#             https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
#             (snapshot consistent with Jan 2017 constituents)
#
# Secondary : MSCI GICS® official methodology document
#             https://www.msci.com/indexes/index-resources/gics
#
# Tertiary  : S&P Dow Jones Indices GICS landing page
#             https://www.spglobal.com/spdji/en/landing/topic/gics/
#
# Notes
# -----
# * Sectors reflect GICS classifications as of 3 Jan 2017, the dataset start.
#   The Sep 2018 GICS reclassification (which renamed "Telecommunication
#   Services" → "Communication Services" and moved media/internet companies
#   into it) is NOT applied here, so GOOG/GOOGL, NFLX, DIS, EA, ATVI, TTWO
#   etc. remain under Communication Services only if they were already there;
#   others that moved in 2018 are kept in their 2017 sectors.
# * Tickers that did not exist for the full 2017-2023 window (ABMD, CTVA,
#   DOW, FOX, FOXA, IR) have been excluded — these are the 6 NaN tickers
#   removed in extract_distance_matrix().
# * PKG (Packaging Corp of America) is classified as Materials per GICS
#   sub-industry "Paper & Plastic Packaging Products & Materials".
# ─────────────────────────────────────────────────────────────────────────────

_GICS_SECTOR_MAP: dict[str, list[str]] = {
    # 18 tickers
    "Communication Services": [
        "T", "VZ", "TMUS", "CMCSA", "CHTR", "DISH",
        "GOOG", "GOOGL", "NFLX", "ATVI", "EA", "TTWO",
        "DIS", "NWS", "NWSA", "IPG", "OMC", "LYV",
    ],
    # 63 tickers
    "Consumer Discretionary": [
        "AMZN", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX",
        "BKNG", "MAR", "HLT", "GM", "F", "ORLY", "AZO", "ROST",
        "DHI", "LEN", "PHM", "NVR", "BBY", "TGT", "EXPE", "YUM",
        "CMG", "DRI", "LVS", "WYNN", "MGM", "CCL", "RCL", "NCLH",
        "HAS", "HOG", "BWA", "APTV", "LEG", "MHK", "NWL",
        "TPR", "PVH", "RL", "VFC", "GPS", "KSS", "JWN", "M",
        "KMX", "CPRI", "LKQ", "TSCO", "ULTA", "GRMN", "WHR",
        "UA", "UAA", "CPRT", "AAP", "DG", "DLTR", "EBAY",
        "GPC", "HBI", "HRB",
    ],
    # 36 tickers
    "Consumer Staples": [
        "PG", "KO", "PEP", "WMT", "COST", "PM", "MO", "MDLZ",
        "CL", "KMB", "GIS", "K", "CPB", "SJM", "HRL", "MKC",
        "CAG", "TSN", "KHC", "HSY", "MNST", "STZ", "TAP",
        "CLX", "CHD", "AVY", "SEE", "ADM", "KR",
        "SYY", "WBA", "CVS", "PRGO", "COTY", "EL", "LW",
    ],
    # 24 tickers
    "Energy": [
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO",
        "OXY", "PXD", "HAL", "DVN", "APA", "HES", "MRO", "BKR",
        "FANG", "NOV", "FTI", "OKE", "KMI", "WMB", "NRG", "HP",
    ],
    # 65 tickers
    "Financials": [
        "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP",
        "BLK", "SCHW", "USB", "PNC", "TFC", "COF", "BK", "STT",
        "MTB", "RF", "KEY", "HBAN", "CMA", "ZION", "CFG", "FITB",
        "SYF", "DFS", "AIG", "PRU", "MET", "AFL", "ALL", "TRV",
        "CB", "PGR", "AJG", "MMC", "AON", "CINF", "HIG", "GL",
        "LNC", "UNM", "PFG", "RE", "RJF", "IVZ", "TROW", "BEN",
        "NDAQ", "ICE", "CME", "CBOE", "MKTX", "SPGI", "MCO",
        "MSCI", "VRSK", "FIS", "GPN", "PYPL", "MA", "V",
        "AMP", "L", "WRB", "AIZ",
    ],
    # 51 tickers
    "Health Care": [
        "JNJ", "UNH", "LLY", "ABBV", "PFE", "MRK", "TMO", "ABT",
        "DHR", "BMY", "AMGN", "GILD", "ISRG", "MDT", "SYK", "BSX",
        "EW", "ZBH", "HOLX", "BAX", "BDX", "COO", "TFX", "IDXX",
        "VRTX", "REGN", "BIIB", "ILMN", "INCY", "HCA",
        "CNC", "CI", "HUM", "DVA", "DGX", "LH", "IQV",
        "PKI", "A", "MTD", "WAT", "RMD", "HSIC", "XRAY", "CAH",
        "MCK", "ABC", "ALGN", "STE", "UHS", "ZTS",
    ],
    # 64 tickers
    "Industrials": [
        "HON", "UPS", "LMT", "BA", "GE", "CAT", "DE",
        "MMM", "EMR", "ETN", "ITW", "ROK", "PH", "DOV", "AME",
        "FTV", "IEX", "SWK", "SNA", "GD", "NOC", "HII",
        "TXT", "LHX", "LDOS", "J", "JCI",
        "XYL", "PNR", "ALLE", "MAS", "AOS", "FLS", "TDG",
        "WAB", "NSC", "UNP", "CSX", "JBHT", "ODFL", "EXPD", "CHRW",
        "FDX", "UAL", "DAL", "LUV", "ALK", "AAL", "GWW", "FAST",
        "CTAS", "ROL", "PAYC", "PAYX", "RHI", "JKHY",
        "RSG", "WM", "ROP", "URI", "CMI", "EFX", "PCAR", "PWR",
    ],
    # 56 tickers
    "Information Technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "INTC",
        "CSCO", "QCOM", "TXN", "ACN", "IBM", "AMAT", "LRCX", "KLAC",
        "MU", "STX", "WDC", "NTAP", "HPQ", "HPE", "CDW",
        "ADBE", "CRM", "NOW", "INTU", "ADSK", "SNPS", "CDNS", "ANSS",
        "FTNT", "CTSH", "IT", "DXC", "JNPR", "ANET", "SWKS", "QRVO",
        "MSI", "KEYS", "TEL", "APH", "GLW", "FFIV", "IPGP",
        "AMD", "NTRS", "BR", "ADP", "AKAM", "ADI", "MCHP",
        "VRSN", "WU", "XRX", "ZBRA", "FLT",
    ],
    # 24 tickers
    "Materials": [
        "LIN", "APD", "SHW", "ECL", "FCX", "NEM", "NUE", "ALB",
        "DD", "EMN", "CE", "LYB", "PPG", "VMC", "MLM",
        "CF", "FMC", "MOS", "IFF", "PKG", "IP",
        "WRK", "AMCR", "WY",
    ],
    # 29 tickers
    "Real Estate": [
        "AMT", "PLD", "CCI", "EQIX", "PSA", "SBAC", "DLR", "O",
        "WELL", "VTR", "PEAK", "EQR", "AVB", "ESS", "MAA", "UDR",
        "AIV", "HST", "BXP", "SLG", "VNO", "KIM", "REG", "FRT",
        "SPG", "EXR", "IRM", "ARE", "CBRE",
    ],
    # 27 tickers
    "Utilities": [
        "NEE", "DUK", "SO", "D", "EXC", "AEP", "XEL", "SRE",
        "ED", "ES", "EIX", "PEG", "ETR", "FE", "PPL", "CMS",
        "CNP", "NI", "AES", "DTE", "LNT", "ATO", "EVRG", "AWK",
        "WEC", "PNW", "AEE",
    ],
}

GICS_SECTOR_ORDER: list[str] = sorted(_GICS_SECTOR_MAP.keys())


def reorder_by_gics(
    distance_matrix: np.ndarray,
    tickers: list[str] | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Reorder a (T, N, N) or (N, N) distance matrix so that stocks are grouped
    by GICS sector.  Within each sector stocks appear in the same relative
    order as in the original ticker list.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Shape (T, 457, 457) or (457, 457).
    tickers : list[str] | None
        The 457 ticker labels corresponding to axis-1 / axis-2.
        Defaults to SP500_TICKERS_457.

    Returns
    -------
    reordered_matrix : np.ndarray
    reordered_tickers : list[str]
    sector_labels : list[str]
        Per-stock sector string, aligned with reordered_tickers.
    """
    if tickers is None:
        tickers = SP500_TICKERS_457

    n = len(tickers)
    ndim = distance_matrix.ndim
    if ndim == 2:
        assert distance_matrix.shape == (n, n)
    elif ndim == 3:
        assert distance_matrix.shape[1] == n and distance_matrix.shape[2] == n
    else:
        raise ValueError(f"distance_matrix must be 2-D or 3-D, got {ndim}-D")

    ticker_to_sector: dict[str, str] = {}
    for sector in GICS_SECTOR_ORDER:
        for t in _GICS_SECTOR_MAP[sector]:
            ticker_to_sector[t] = sector

    missing_tickers = [t for t in tickers if t not in ticker_to_sector]
    if missing_tickers:
        raise ValueError(
            f"The following tickers have no GICS sector mapping: {missing_tickers}\n"
            "Update _GICS_SECTOR_MAP in extract_distance_matrices.py."
        )

    ticker_index = {t: i for i, t in enumerate(tickers)}
    perm: list[int] = []
    reordered_tickers: list[str] = []
    sector_labels: list[str] = []

    for sector in GICS_SECTOR_ORDER:
        sector_tickers_in_data = [
            t for t in tickers if ticker_to_sector.get(t) == sector
        ]
        for t in sector_tickers_in_data:
            perm.append(ticker_index[t])
            reordered_tickers.append(t)
            sector_labels.append(sector)

    perm_arr = np.array(perm, dtype=np.intp)

    if ndim == 2:
        reordered = distance_matrix[np.ix_(perm_arr, perm_arr)]
    else:
        reordered = distance_matrix[:, perm_arr, :][:, :, perm_arr]

    return reordered, reordered_tickers, sector_labels


def get_gics_sector_boundaries(sector_labels: list[str]) -> list[tuple[str, int, int]]:
    """
    Given the per-stock sector_labels list returned by reorder_by_gics(),
    compute the start and end index of each GICS sector block.

    Returns
    -------
    list of (sector_name, start_idx, end_idx) tuples  (end_idx is exclusive)
    """
    boundaries: list[tuple[str, int, int]] = []
    current_sector = sector_labels[0]
    start = 0
    for i, s in enumerate(sector_labels[1:], start=1):
        if s != current_sector:
            boundaries.append((current_sector, start, i))
            current_sector = s
            start = i
    boundaries.append((current_sector, start, len(sector_labels)))
    return boundaries


def build_patch_sector_ids(
    sector_labels: list[str],
    patch_size: int = 16,
    img_size: int = 457,
) -> torch.Tensor:
    """
    Build a (N,) integer tensor mapping each image patch to its dominant
    GICS sector index.  Required by SectorGPSA in transformer.py.

    The 457×457 GICS-reordered distance matrix is padded to padded_size
    (= ceil(457/patch_size) * patch_size) before patchification.  Each
    patch covers patch_size stocks along both the row and column axes.
    The dominant sector for a patch is the GICS sector that covers the
    majority of the stock-rows within that patch.

    Parameters
    ----------
    sector_labels : list[str]
        Per-stock sector string in GICS order, as returned by
        reorder_by_gics().  Length must be 457.
    patch_size    : int
        Patch size used by the model (default 16).
    img_size      : int
        Original image size before padding (default 457).

    Returns
    -------
    sector_ids : torch.Tensor  shape (N,) dtype=torch.long
        Integer sector index for each patch.  N = grid_h * grid_w.
        Sector index follows GICS_SECTOR_ORDER (alphabetical).

    Notes
    -----
    Because the matrix is symmetric and GICS-reordered, a patch at grid
    position (r, c) covers stocks [r*p, (r+1)*p) on the row axis and
    [c*p, (c+1)*p) on the column axis.  For the sector ID we use the
    row axis only (stocks correspond to rows after GICS reordering),
    which gives a consistent spatial sector map aligned with the
    block-diagonal structure.

    Padded rows (beyond stock index 456) inherit the last stock's sector,
    keeping the boundary patches consistent.
    """
    import math

    assert len(sector_labels) == img_size, (
        f"sector_labels length {len(sector_labels)} != img_size {img_size}"
    )

    padded_size = math.ceil(img_size / patch_size) * patch_size
    grid        = padded_size // patch_size   # grid_h == grid_w
    N           = grid * grid

    # Build per-pixel (per-stock-row) sector index, padded to padded_size
    sector_to_idx = {s: i for i, s in enumerate(GICS_SECTOR_ORDER)}
    # Map each stock index → integer sector index
    stock_sector_idx = [sector_to_idx[s] for s in sector_labels]
    # Pad with the last stock's sector for reflect-padded rows beyond img_size
    last_sector_idx = stock_sector_idx[-1]
    padded_stock_sector = stock_sector_idx + [last_sector_idx] * (padded_size - img_size)

    # For each patch row r, find the majority sector among stocks
    # in rows [r*patch_size, (r+1)*patch_size)
    patch_row_sector = []
    for r in range(grid):
        row_stocks = padded_stock_sector[r * patch_size: (r + 1) * patch_size]
        # Majority vote (Counter-style)
        from collections import Counter
        majority_sector = Counter(row_stocks).most_common(1)[0][0]
        patch_row_sector.append(majority_sector)

    # Each patch (r, c) inherits the sector of its row
    sector_ids = []
    for r in range(grid):
        for c in range(grid):
            sector_ids.append(patch_row_sector[r])

    return torch.tensor(sector_ids, dtype=torch.long)
