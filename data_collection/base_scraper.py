from abc import ABC, abstractmethod
from typing import List

import requests
from bs4 import BeautifulSoup

from data_collection.data import Document


class BaseWebScraper(ABC):

    @abstractmethod
    def scrape(self) -> List[Document]:
        """Scrape data from web"""
        raise NotImplementedError

    @classmethod
    def get_soup(cls, url: str) -> BeautifulSoup:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup
