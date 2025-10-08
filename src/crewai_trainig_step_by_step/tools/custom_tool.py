import json
import os
from typing import Optional, Type

import requests
from crewai.tools import BaseTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class SerperScrapeInput(BaseModel):
    """Input schema for Serper Scrape Tool."""

    url: str = Field(..., description="The URL to scrape content from")
    include_markdown: bool = Field(
        default=True,
        description="Whether to include markdown formatting in the response",
    )


class SerperScrapeTool(BaseTool):
    name: str = "Serper Web Scraper"
    description: str = (
        "Scrapes web content from any URL using the Serper API. "
        "Extracts clean, readable content with optional markdown formatting. "
        "Use this tool when you need to get the full content of a webpage for analysis."
    )
    args_schema: Type[BaseModel] = SerperScrapeInput

    def _run(self, url: str, include_markdown: bool = True) -> str:
        """
        Execute the Serper API scraping operation.

        Args:
            url: The URL to scrape
            include_markdown: Whether to include markdown formatting

        Returns:
            The scraped content as a string
        """
        try:
            # Get API key from environment
            api_key = os.getenv("SERPER_API_KEY")
            if not api_key:
                return "Error: SERPER_API_KEY not found in environment variables. Please add it to your .env file."

            # Prepare the API request
            serper_url = "https://scrape.serper.dev"

            # Ensure URL has proper protocol
            if not url.startswith(("http://", "https://")):
                url = f"https://{url}"

            payload = json.dumps({"url": url, "includeMarkdown": include_markdown})

            headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

            # Make the API request
            response = requests.post(serper_url, headers=headers, data=payload)
            response.raise_for_status()

            # Parse the response
            data = response.json()

            # Extract the content
            if "text" in data:
                content = data["text"]

                # Add metadata
                result = f"**Scraped from:** {url}\n"
                result += f"**Markdown formatting:** {'Enabled' if include_markdown else 'Disabled'}\n"
                result += f"**Content length:** {len(content)} characters\n\n"
                result += "**Content:**\n" + content

                return result
            else:
                return f"No content found for URL: {url}. Response: {data}"

        except requests.exceptions.RequestException as e:
            return f"Network error while scraping {url}: {str(e)}"
        except json.JSONDecodeError as e:
            return f"JSON parsing error from Serper API: {str(e)}"
        except Exception as e:
            return f"Unexpected error while scraping {url}: {str(e)}"


# Keep the original tool for backward compatibility
class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""

    argument: str = Field(..., description="Description of the argument.")


class MyCustomTool(BaseTool):
    name: str = "Legacy Custom Tool"
    description: str = "Legacy tool for backward compatibility. Use SerperScrapeTool instead for web scraping."
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "This is the legacy tool. Consider using SerperScrapeTool for web scraping tasks."
