from unittest.mock import Mock, patch

import pytest
from bs4 import BeautifulSoup

from ..indexer import SitemapParser, WordPressScraperIndexer


@pytest.fixture
def indexer():
    return WordPressScraperIndexer("https://localclubhouse.com")


def test_get_sitemap_url(indexer):
    assert (
        indexer.create_sitemap_url("https://localclubhouse.com")
        == "https://localclubhouse.com/sitemap.xml"
    )


@pytest.fixture
def sitemap_parser():
    return SitemapParser()


def test_init_default_values():
    parser = SitemapParser()
    assert parser.delay == 1.0
    assert "User-Agent" in parser.session.headers


def test_init_custom_values():
    custom_agent = "CustomBot/1.0"
    custom_delay = 2.0
    parser = SitemapParser(user_agent=custom_agent, delay=custom_delay)
    assert parser.delay == custom_delay
    assert parser.session.headers["User-Agent"] == custom_agent


def test_parse_xml(sitemap_parser):
    content = b"<urlset><url><loc>http://example.com</loc></url></urlset>"
    soup = sitemap_parser.parse_xml(content)
    assert isinstance(soup, BeautifulSoup)
    assert soup.find("loc").text == "http://example.com"


def test_extract_urls(sitemap_parser):
    xml = "<urlset><url><loc>http://example.com/1</loc></url><url><loc>http://example.com/2</loc></url></urlset>"
    soup = BeautifulSoup(xml, "xml")
    urls = sitemap_parser.extract_urls(soup)
    assert urls == ["http://example.com/1", "http://example.com/2"]


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://example.com/sitemap.xml", True),
        ("http://example.com/page.html", False),
    ],
)
def test_is_sitemap(sitemap_parser, url, expected):
    assert sitemap_parser.is_sitemap(url) == expected


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://example.com/page.html", True),
        ("http://example.com/image.jpg", False),
        ("http://example.com/document.pdf", False),
    ],
)
def test_is_html_page(sitemap_parser, url, expected):
    assert sitemap_parser.is_html_page(url) == expected


@patch("indexer.SitemapParser.fetch_content")
@patch("indexer.SitemapParser.parse_xml")
def test_get_all_page_urls(mock_parse_xml, mock_fetch_content, sitemap_parser):
    mock_fetch_content.return_value = b"<xml>content</xml>"
    mock_soup = Mock()
    mock_parse_xml.return_value = mock_soup

    mock_soup.find_all.return_value = [
        Mock(text="http://example.com/sitemap2.xml"),
        Mock(text="http://example.com/page1.html"),
        Mock(text="http://example.com/page2.html"),
        Mock(text="http://example.com/image.jpg"),
    ]

    urls = sitemap_parser.get_all_page_urls("http://example.com/sitemap.xml")
    assert urls == {"http://example.com/page1.html", "http://example.com/page2.html"}
    assert (
        mock_fetch_content.call_count == 2
    )  # Once for main sitemap, once for nested sitemap


if __name__ == "__main__":
    pytest.main()
