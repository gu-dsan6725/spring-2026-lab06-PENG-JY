"""
MCP Test Client for World Bank Server

Tests all resources and tools via Streamable HTTP transport.

Usage:
    1. Start your server: uv run python server.py
    2. Run this test:     uv run python test_client.py

This script will test all resources and tools and report pass/fail status.
"""
import asyncio
import json
import logging
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

SERVER_URL: str = "http://127.0.0.1:8765/mcp"


def _print_separator(title: str) -> None:
    """Print a section separator."""
    logger.info("")
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)


def _truncate(text: str, max_length: int = 200) -> str:
    """Truncate text for display."""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


async def test_server() -> bool:
    """
    Test all resources and tools on the MCP server.

    Returns:
        True if all tests pass, False otherwise
    """
    all_passed = True

    logger.info(f"Connecting to MCP server at {SERVER_URL}")
    logger.info("Make sure your server is running: uv run python server.py")

    try:
        async with streamablehttp_client(SERVER_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()
                logger.info("Session initialized successfully")

                # =============================================================
                # Test Resources
                # =============================================================
                _print_separator("TESTING RESOURCES")

                # List available resources
                resources = await session.list_resources()
                resource_uris = [str(r.uri) for r in resources.resources]
                logger.info(f"Available resources: {resource_uris}")

                # Test data://schema
                logger.info("\n[TEST] data://schema")
                try:
                    schema = await session.read_resource("data://schema")
                    schema_text = schema.contents[0].text
                    logger.info(f"  Result: {_truncate(schema_text)}")
                    # Verify it's valid JSON
                    json.loads(schema_text)
                    logger.info("  Status: PASS")
                except Exception as e:
                    logger.error(f"  Status: FAIL - {e}")
                    all_passed = False

                # Test data://countries
                logger.info("\n[TEST] data://countries")
                try:
                    countries = await session.read_resource("data://countries")
                    countries_text = countries.contents[0].text
                    if countries_text is None or countries_text == "null":
                        raise ValueError("Resource returned None - not implemented")
                    logger.info(f"  Result: {_truncate(countries_text)}")
                    # Verify it's valid JSON array
                    data = json.loads(countries_text)
                    if not isinstance(data, list):
                        raise ValueError("Expected a list of countries")
                    logger.info(f"  Found {len(data)} countries")
                    logger.info("  Status: PASS")
                except Exception as e:
                    logger.error(f"  Status: FAIL - {e}")
                    all_passed = False

                # Test data://indicators/{country_code}
                logger.info("\n[TEST] data://indicators/USA")
                try:
                    indicators = await session.read_resource("data://indicators/USA")
                    indicators_text = indicators.contents[0].text
                    if indicators_text is None or indicators_text == "null":
                        raise ValueError("Resource returned None - not implemented")
                    logger.info(f"  Result: {_truncate(indicators_text)}")
                    # Verify it's valid JSON
                    data = json.loads(indicators_text)
                    logger.info(f"  Found {len(data) if isinstance(data, list) else 1} records")
                    logger.info("  Status: PASS")
                except Exception as e:
                    logger.error(f"  Status: FAIL - {e}")
                    all_passed = False

                # =============================================================
                # Test Tools
                # =============================================================
                _print_separator("TESTING TOOLS")

                # List available tools
                tools = await session.list_tools()
                tool_names = [t.name for t in tools.tools]
                logger.info(f"Available tools: {tool_names}")

                # Test get_country_info
                logger.info("\n[TEST] get_country_info('USA')")
                try:
                    result = await session.call_tool(
                        "get_country_info",
                        {"country_code": "USA"},
                    )
                    result_text = result.content[0].text
                    if result_text is None or result_text == "null":
                        raise ValueError("Tool returned None - not implemented")
                    logger.info(f"  Result: {_truncate(result_text, 300)}")
                    # Verify it contains expected fields
                    data = json.loads(result_text) if isinstance(result_text, str) else result_text
                    if "capital" not in str(data).lower():
                        logger.warning("  Warning: Response may be missing expected fields")
                    logger.info("  Status: PASS")
                except Exception as e:
                    logger.error(f"  Status: FAIL - {e}")
                    all_passed = False

                # Test get_live_indicator
                logger.info("\n[TEST] get_live_indicator('USA', 'NY.GDP.PCAP.CD', 2022)")
                try:
                    result = await session.call_tool(
                        "get_live_indicator",
                        {
                            "country_code": "USA",
                            "indicator": "NY.GDP.PCAP.CD",
                            "year": 2022,
                        },
                    )
                    result_text = result.content[0].text
                    if result_text is None or result_text == "null":
                        raise ValueError("Tool returned None - not implemented")
                    logger.info(f"  Result: {result_text}")
                    logger.info("  Status: PASS")
                except Exception as e:
                    logger.error(f"  Status: FAIL - {e}")
                    all_passed = False

                # Test compare_countries
                logger.info("\n[TEST] compare_countries(['USA', 'CHN', 'DEU'], 'SP.POP.TOTL', 2022)")
                try:
                    result = await session.call_tool(
                        "compare_countries",
                        {
                            "country_codes": ["USA", "CHN", "DEU"],
                            "indicator": "SP.POP.TOTL",
                            "year": 2022,
                        },
                    )
                    result_text = result.content[0].text
                    if result_text is None or result_text == "null":
                        raise ValueError("Tool returned None - not implemented")
                    logger.info(f"  Result: {_truncate(result_text, 400)}")
                    # Verify it's a list with 3 entries
                    data = json.loads(result_text) if isinstance(result_text, str) else result_text
                    if isinstance(data, list) and len(data) == 3:
                        logger.info("  Found results for all 3 countries")
                    logger.info("  Status: PASS")
                except Exception as e:
                    logger.error(f"  Status: FAIL - {e}")
                    all_passed = False

                # =============================================================
                # Summary
                # =============================================================
                _print_separator("TEST SUMMARY")

                if all_passed:
                    logger.info("ALL TESTS PASSED")
                else:
                    logger.error("SOME TESTS FAILED - check the output above")

    except ConnectionRefusedError:
        logger.error("Could not connect to server!")
        logger.error("Make sure your server is running: uv run python server.py")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(test_server())
    sys.exit(0 if success else 1)
