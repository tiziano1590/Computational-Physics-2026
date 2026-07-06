import requests
import re
import zipfile
from pathlib import Path
from urllib.parse import urljoin
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

# Reference paper that inspired the workflow and the optional ESSA-style
# extension: Elsevier PII S0019103525002222 (Icarus). Not part of the data
# manifest — it is a paper, not a downloadable dataset.
DOWNLOAD_MANIFEST = [
    {
        "key": "wac_global_product_page_016p",
        "category": "imagery",
        "access_mode": "page_regex",
        "access_url_or_page": "https://data.lroc.im-ldi.com/lroc/view_rdr_product/WAC_GLOBAL_E000N0000_016P",
        "match_or_direct_url": "WAC_GLOBAL_E000N0000_016P\\.TIF$",
        "purpose": "Default light-weight LRO morphology raster for quick demos and CPU-friendly runs.",
        "notes": "Official LROC product page; regex extracts the GeoTIFF."
    },
    {
        "key": "wac_global_product_page_064p",
        "category": "imagery",
        "access_mode": "page_regex",
        "access_url_or_page": "https://data.lroc.im-ldi.com/lroc/view_rdr_product/WAC_GLOBAL_E000N0000_064P",
        "match_or_direct_url": "WAC_GLOBAL_E000N0000_064P\\.TIF$",
        "purpose": "Higher-detail LRO morphology raster for research runs.",
        "notes": "Official LROC product naming pattern from the WAC Global Morphologic Map page."
    },
    {
        "key": "lunar_pit_locations_page",
        "category": "vector_labels",
        "access_mode": "page_regex",
        "access_url_or_page": "https://data.lroc.im-ldi.com/lroc/view_rdr/SHAPEFILE_LUNAR_PIT_LOCATIONS",
        "match_or_direct_url": "LUNAR_PIT_LOCATIONS\\.CSV$|LUNAR_PIT_LOCATIONS_180\\.ZIP$",
        "purpose": "Pit/skylight labels, diameters, and metadata.",
        "notes": "Notebook uses the CSV for weak labels and the shapefile if present."
    },
    {
        "key": "wrinkle_ridges_page",
        "category": "vector_labels",
        "access_mode": "page_regex",
        "access_url_or_page": "https://data.lroc.im-ldi.com/lroc/view_rdr/SHAPEFILE_WRINKLE_RIDGES",
        "match_or_direct_url": "WRINKLE_RIDGES_180\\.ZIP$",
        "purpose": "Wrinkle ridge line labels.",
        "notes": "Official LROC vector layer."
    },
    {
        "key": "lobate_scarps_page",
        "category": "vector_labels",
        "access_mode": "page_regex",
        "access_url_or_page": "https://data.lroc.im-ldi.com/lroc/view_rdr/SHAPEFILE_LOBATE_SCARPS",
        "match_or_direct_url": "LOBATE_SCARPS_180\\.ZIP$",
        "purpose": "Lobate scarp line labels.",
        "notes": "Official LROC vector layer."
    },
    {
        "key": "imp_locations_page",
        "category": "vector_labels",
        "access_mode": "page_regex",
        "access_url_or_page": "https://data.lroc.im-ldi.com/lroc/view_rdr/SHAPEFILE_LUNAR_IMP_LOCATIONS",
        "match_or_direct_url": "LUNAR_IMP_LOCATIONS_180\\.ZIP$",
        "purpose": "Irregular mare patch locations.",
        "notes": "Official LROC vector layer."
    },
    {
        "key": "anthropogenic_objects_page",
        "category": "vector_labels",
        "access_mode": "page_regex",
        "access_url_or_page": "https://data.lroc.im-ldi.com/lroc/view_rdr/SHAPEFILE_ANTHROPOGENIC_OBJECTS",
        "match_or_direct_url": "ANTHROPOGENIC_OBJECTS_180\\.ZIP$",
        "purpose": "Apollo landing sites and other anthropogenic lunar objects.",
        "notes": "Official LROC vector layer."
    },
    {
        "key": "moon_crater_database_page",
        "category": "vector_labels",
        "access_mode": "page_regex",
        "access_url_or_page": "https://astrogeology.usgs.gov/search/map/moon_crater_database_v1_robbins",
        "match_or_direct_url": "lunar_crater_database_robbins_2018$",
        "purpose": "Global lunar crater database for crater masks.",
        "notes": "Official USGS Astrogeology landing page; the package contains the crater table and GIS files."
    },
    {
        "key": "lunar_sinuous_rilles_table1",
        "category": "vector_labels",
        "access_mode": "direct",
        "access_url_or_page": "https://www.lpi.usra.edu/lunar/rilles/Hurwitzetal_PSS2012_LunarSinuousRillesSurvey_Table1_UPDATED.xlsx",
        "match_or_direct_url": "direct xlsx",
        "purpose": "Sinuous rille atlas Table 1.",
        "notes": "Used as an optional proxy label source for lava-tube-related surface expressions."
    },
    {
        "key": "lunar_sinuous_rilles_table2",
        "category": "vector_labels",
        "access_mode": "direct",
        "access_url_or_page": "https://www.lpi.usra.edu/lunar/rilles/Hurwitzetal_PSS2012_LunarSinuousRillesSurvey_Table2_UPDATED.xlsx",
        "match_or_direct_url": "direct xlsx",
        "purpose": "Sinuous rille atlas Table 2.",
        "notes": "Used as an optional proxy label source for lava-tube-related surface expressions."
    },
    # NOTE: the crater CSV (lunar_crater_database_robbins_2018.csv) is not a
    # separate download — it ships inside the USGS bundle above and
    # label_loader.load_all_labels() locates it by filename in raw_dir.
    {
        "key": "essa_zenodo_page",
        "category": "optional_reference",
        "access_mode": "direct",
        "access_url_or_page": "https://zenodo.org/records/15438463/files/ESSA_shapefiles.zip?download=1",
        "match_or_direct_url": "direct zip",
        "purpose": "Optional Marius Hills rille shapefile and ESSA detections from the authors.",
        "notes": "Use only if you want paper-aligned proxy labels or comparison layers."
    },
    {
        "key": "apollo_coordinates_nasa",
        "category": "optional_reference",
        "access_mode": "direct",
        "access_url_or_page": "https://nssdc.gsfc.nasa.gov/planetary/lunar/lunar_sites.html",
        "match_or_direct_url": "reference page",
        "purpose": "Cross-check for Apollo landing site coordinates.",
        "notes": "Optional cross-check; the notebook uses the more complete LROC anthropogenic layer by default."
    },
    {
        "key": "apollo11_controlled_mosaic_page",
        "category": "optional_highres",
        "access_mode": "direct",
        "access_url_or_page": "https://data.lroc.im-ldi.com/lroc/view_rdr_product/NAC_ROI_APOLLO11HIA_E010N0234",
        "match_or_direct_url": "native TIF link on page",
        "purpose": "Optional meter-scale controlled mosaic example for Apollo 11.",
        "notes": "Used in the ESSA-style high-resolution extension."
    },
    {
        "key": "apollo15_controlled_mosaic_page",
        "category": "optional_highres",
        "access_mode": "direct",
        "access_url_or_page": "https://data.lroc.im-ldi.com/lroc/view_rdr/NAC_ROI_APOLLO15LOB",
        "match_or_direct_url": "native/5 m/20 m products linked from page",
        "purpose": "Optional meter-scale controlled mosaic example for Apollo 15 and Hadley Rille.",
        "notes": "Used in the ESSA-style high-resolution extension."
    },
    # NAC High-Resolution Products (1-5m resolution)
    {
        "key": "nac_apollo11",
        "category": "imagery",
        "access_mode": "direct",
        "access_url_or_page": "https://data.lroc.im-ldi.com/lroc/view_rdr_product/NAC_ROI_APOLLO11HIA_E010N0234",
        "match_or_direct_url": "direct TIF",
        "purpose": "Apollo 11 Hadley Rille area - High-resolution rille mapping.",
        "notes": "NAC Red channel - 1.65m/pixel resolution"
    }
]

def request_text(url: str, timeout: int = 60) -> str:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text

def extract_link_from_page(page_url: str, pattern: str) -> str:
    html = request_text(page_url)
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    for href in hrefs:
        full = urljoin(page_url, href)
        if re.search(pattern, full, flags=re.IGNORECASE):
            return full
    raise FileNotFoundError(f'Could not find a download link matching {pattern!r} on {page_url}')

def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f'Using cached file: {dest.name}')
        return dest
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(dest, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=dest.name) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))
    return dest

def extract_any(path: Path, out_dir: Path | None = None) -> Path:
    path = Path(path)
    if out_dir is None:
        out_dir = path.with_suffix('')
        if out_dir == path:
            out_dir = path.parent / (path.name + '_extracted')
    out_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(out_dir)
        return out_dir
    return path

def resolve_row(row: dict, preferred_scale: str = '064P', fallback_scale: str = '016P') -> dict:
    key = row['key']
    mode = row['access_mode']
    page = row['access_url_or_page']
    match = row['match_or_direct_url']

    if key.startswith('wac_global_product_page_'):
        return row

    if key in ['nac_hadley_rille_027', 'nac_hadley_rille_028', 'nac_hadley_rille_026',
               'nac_apollo11', 'nac_marius_hills', 'nac_copernicus', 'nac_tycho']:
        # NAC products - use direct page URL
        return {**row, 'resolved_url': page}

    if mode == 'direct':
        return {**row, 'resolved_url': page}

    if mode == 'local_file':
        return {**row, 'resolved_url': page}

    resolved = extract_link_from_page(page, match)
    return {**row, 'resolved_url': resolved}

def resolve_wac_tif(preferred_scale: str = '064P', fallback_scale: str = '016P') -> str:
    page_template = 'https://data.lroc.im-ldi.com/lroc/view_rdr_product/WAC_GLOBAL_E000N0000_{scale}'
    regex_template = r'WAC_GLOBAL_E000N0000_{scale}\.TIF$'
    for scale in [preferred_scale, fallback_scale]:
        page_url = page_template.format(scale=scale)
        pattern = regex_template.format(scale=scale)
        try:
            return extract_link_from_page(page_url, pattern)
        except Exception as exc:
            logger.warning(f'WAC resolve failed for {scale}: {exc}')
    raise FileNotFoundError('Could not resolve any WAC GeoTIFF link.')

def prepare_dataset(data_dir: str):
    raw_dir = Path(data_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    resolved = {}

    # Resolve and download WAC
    try:
        wac_url = resolve_wac_tif()
        wac_dest = raw_dir / Path(wac_url).name
        download_file(wac_url, wac_dest)
        resolved['wac_global_tif'] = str(wac_dest)
    except Exception as e:
        logger.error(f"Failed to resolve/download WAC: {e}")

    for row in DOWNLOAD_MANIFEST:
        if row['key'].startswith('wac_global_product_page_'):
            continue
        try:
            resolved_row = resolve_row(row)
            url = resolved_row.get('resolved_url')
            if not url:
                continue

            target_name = Path(url).name or row['key']
            dest = raw_dir / target_name

            if row['access_mode'] != 'local_file':
                download_file(url, dest)

            if '.zip' in target_name.lower():
                extract_any(dest, raw_dir / f"{target_name}_extracted")

            # Extract CSV from crater bundle
            if 'crater' in row['key'] and 'csv' not in dest.name.lower():
                csv_dest = raw_dir / f"{target_name}.csv"
                if not csv_dest.exists():
                    # Look for the CSV in the bundle
                    csv_path = dest.parent / "bundle" / "data" / "lunar_crater_database_robbins_2018.csv"
                    if csv_path.exists():
                        csv_dest.write_bytes(csv_path.read_bytes())
                        resolved[f"{row['key']}_csv"] = str(csv_dest)
                        logger.info(f"Extracted CSV: {row['key']}_csv")

            resolved[row['key']] = str(dest)
            logger.info(f"Successfully prepared {row['key']}")
        except Exception as e:
            logger.error(f"Failed to prepare {row['key']}: {e}")

    return resolved
