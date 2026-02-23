"""
Week 4 Lab: World Bank Data MCP Server

An MCP server that exposes:
- Resources: Local World Bank indicator data from CSV
- Tools: Live data from REST Countries and World Bank APIs

Transport: Streamable HTTP on port 8765
"""
import json
import logging
from pathlib import Path
from typing import Optional

import httpx
import polars as pl
from mcp.server.fastmcp import FastMCP


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_FILE: Path = Path(__file__).parent / "data" / "world_bank_indicators.csv"
HOST: str = "127.0.0.1"
PORT: int = 8765

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP(
    "world-bank-server",
    host=HOST,
    port=PORT,
)


# =============================================================================
# PRIVATE HELPER FUNCTIONS
# =============================================================================

def _load_data() -> pl.DataFrame:
    """Load the World Bank indicators CSV file."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")
    return pl.read_csv(DATA_FILE)


def _fetch_rest_countries(country_code: str) -> dict:
    """Fetch country info from REST Countries API."""
    url = f"https://restcountries.com/v3.1/alpha/{country_code}"
    with httpx.Client(timeout=30.0) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json()[0]


def _fetch_world_bank_indicator(
    country_code: str,
    indicator: str,
    year: Optional[int] = None,
) -> list:
    """Fetch indicator from World Bank API."""
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}"
    params = {"format": "json", "per_page": 100}
    if year:
        params["date"] = str(year)

    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if len(data) < 2 or not data[1]:
            return []
        return data[1]


# =============================================================================
# PART 1: RESOURCES (Local Data)
# =============================================================================

@mcp.resource("data://schema")
def get_schema() -> str:
    """
    Return the schema of the World Bank dataset.

    This resource is provided as an example - it's already implemented.
    """
    df = _load_data()
    schema_info = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
    return json.dumps(schema_info, indent=2)


@mcp.resource("data://countries")
def get_countries() -> str:
    """
    List all unique countries in the dataset.

    Returns:
        JSON string of unique country codes and names.
        Format: [{"countryiso3code": "USA", "country": "United States"}, ...]
    """
    try:
        df = _load_data()
        result = df.select(["countryiso3code", "country"]).unique()
        return result.write_json()
    except FileNotFoundError as e:
        logger.error(f"Data file missing: {e}")
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error(f"Unexpected error loading countries: {e}")
        return json.dumps({"error": f"Failed to load countries: {str(e)}"})


@mcp.resource("data://indicators/{country_code}")
def get_country_indicators(country_code: str) -> str:
    """
    Get all indicators for a specific country from local data.

    Args:
        country_code: ISO 3166-1 alpha-3 country code (e.g., "USA", "CHN", "DEU")

    Returns:
        JSON string of all indicator records for the given country.
        Returns an error JSON object if the country code is not found.
    """
    try:
        df = _load_data()
        filtered = df.filter(pl.col("countryiso3code") == country_code.upper())
        if filtered.is_empty():
            logger.warning(f"Country code not found in local data: {country_code}")
            return json.dumps({"error": f"Country code '{country_code}' not found in local dataset"})
        return filtered.write_json()
    except FileNotFoundError as e:
        logger.error(f"Data file missing: {e}")
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error(f"Unexpected error fetching indicators for {country_code}: {e}")
        return json.dumps({"error": f"Failed to fetch indicators: {str(e)}"})


# =============================================================================
# PART 2: TOOLS (External APIs)
# =============================================================================

@mcp.tool()
def get_country_info(country_code: str) -> dict:
    """
    Fetch detailed information about a country from REST Countries API.

    Args:
        country_code: ISO 3166-1 alpha-2 or alpha-3 country code (e.g., "US", "USA", "DE")

    Returns:
        Dictionary with country information including name, capital, region,
        subregion, languages, currencies, population, and flag emoji.
    """
    logger.info(f"Fetching country info for: {country_code}")
    try:
        data = _fetch_rest_countries(country_code)
        return {
            "name": data["name"]["common"],
            "capital": data.get("capital", ["N/A"])[0],
            "region": data.get("region", "N/A"),
            "subregion": data.get("subregion", "N/A"),
            "languages": list(data.get("languages", {}).values()),
            "currencies": list(data.get("currencies", {}).keys()),
            "population": data.get("population", 0),
            "flag": data.get("flag", ""),
        }
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.warning(f"Country not found: {country_code}")
            return {"error": f"Country code '{country_code}' not found"}
        logger.error(f"API HTTP error for {country_code}: {e}")
        return {"error": f"API error ({e.response.status_code}): {str(e)}"}
    except httpx.TimeoutException:
        logger.error(f"Request timed out for country: {country_code}")
        return {"error": "Request timed out, please try again"}
    except httpx.RequestError as e:
        logger.error(f"Network error fetching country info for {country_code}: {e}")
        return {"error": f"Network error: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error fetching country info for {country_code}: {e}")
        return {"error": f"Unexpected error: {str(e)}"}


@mcp.tool()
def get_live_indicator(
    country_code: str,
    indicator: str,
    year: int = 2022,
) -> dict:
    """
    Fetch a specific indicator value from the World Bank API.

    Args:
        country_code: ISO 3166-1 alpha-2 or alpha-3 country code
        indicator: World Bank indicator ID (e.g., "NY.GDP.PCAP.CD" for GDP per capita)
        year: Year to fetch data for (default: 2022)

    Returns:
        Dictionary with country, country_name, indicator, indicator_name, year, and value.

    Common indicators:
        - NY.GDP.PCAP.CD: GDP per capita (current US$)
        - SP.POP.TOTL: Total population
        - SP.DYN.LE00.IN: Life expectancy at birth
        - SE.ADT.LITR.ZS: Adult literacy rate
    """
    logger.info(f"Fetching {indicator} for {country_code} in {year}")
    try:
        records = _fetch_world_bank_indicator(country_code, indicator, year)
        if not records:
            logger.warning(f"No records returned for {country_code} / {indicator} / {year}")
            return {"error": f"No data available for {country_code} / {indicator} in {year}"}
        # 找到匹配年份的记录
        for record in records:
            if record.get("date") == str(year):
                return {
                    "country": country_code,
                    "country_name": record.get("country", {}).get("value", "N/A"),
                    "indicator": indicator,
                    "indicator_name": record.get("indicator", {}).get("value", "N/A"),
                    "year": year,
                    "value": record.get("value"),
                }
        logger.warning(f"Year {year} not found in results for {country_code} / {indicator}")
        return {"error": f"No data found for {country_code} / {indicator} in {year}"}
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.warning(f"Indicator or country not found: {country_code} / {indicator}")
            return {"error": f"Country '{country_code}' or indicator '{indicator}' not found"}
        logger.error(f"API HTTP error for {country_code} / {indicator}: {e}")
        return {"error": f"API error ({e.response.status_code}): {str(e)}"}
    except httpx.TimeoutException:
        logger.error(f"Request timed out for {country_code} / {indicator}")
        return {"error": "Request timed out, please try again"}
    except httpx.RequestError as e:
        logger.error(f"Network error fetching indicator {indicator} for {country_code}: {e}")
        return {"error": f"Network error: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error fetching {indicator} for {country_code}: {e}")
        return {"error": f"Unexpected error: {str(e)}"}


@mcp.tool()
def compare_countries(
    country_codes: list[str],
    indicator: str,
    year: int = 2022,
) -> list[dict]:
    """
    Compare an indicator across multiple countries.

    Args:
        country_codes: List of ISO country codes to compare (e.g., ["USA", "CHN", "DEU"])
        indicator: World Bank indicator ID to compare
        year: Year to fetch data for (default: 2022)

    Returns:
        List of dictionaries, one per country, each containing country, country_name,
        indicator, year, and value. Individual failures don't abort the whole request.
    """
    logger.info(f"Comparing {indicator} for countries: {country_codes}")

    if not country_codes:
        logger.warning("compare_countries called with empty country list")
        return [{"error": "No country codes provided"}]

    results = []
    for code in country_codes:
        try:
            result = get_live_indicator(code, indicator, year)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to fetch {indicator} for {code}: {e}")
            results.append({
                "country": code,
                "indicator": indicator,
                "year": year,
                "value": None,
                "error": str(e),
            })
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logger.info(f"Starting World Bank MCP Server on http://{HOST}:{PORT}/mcp")
    logger.info(f"Connect with MCP Inspector or test client at http://{HOST}:{PORT}/mcp")
    logger.info("Press Ctrl+C to stop")
    mcp.run(transport="streamable-http")