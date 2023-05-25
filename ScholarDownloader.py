import os
from scholarly import scholarly
from scidownl import scihub_download
import requests

DIRECTORY = "papers"

class ScholarlyDownloader:
    def __init__(self):
        pass
    
    def search(self, query, limit=1):
        search_query = scholarly.search_pubs(query)
        pub = next(search_query, None)
        papers = []
        
        if pub is None:
            print('No results found.')
            return None
        else:
            for paper in pub:
                papers.append(paper)
            return pub
    
    def download_pdf(self, pub):
        pdf_url = pub['pub_url'] if '.pdf' not in pub['pub_url'] else None
        
        if pdf_url is None:
            print('PDF not found on Google Scholar, trying Sci-Hub...')
            pdf_filename = self.download_private_pub(pub)
            
            if not pdf_filename:
                print('PDF not found on Sci-Hub.')
                return None
            
            return pdf_filename
        
        else: 
            pdf_data = requests.get(pdf_url).content
            pdf_filename = DIRECTORY + "/" + pub["bib"]['title'].replace('/', '-') + '.pdf'
            
            with open(pdf_filename, 'wb') as f:
                f.write(pdf_data)
                
            return pdf_filename
        
    def download_private_pub(self, pub):
        pdf_url = pub['pub_url'] if '.pdf' not in pub['pub_url'] else None
        
        if pdf_url is None:
            return ""
        else:
            scihub_download(pub["pub_url"], out = DIRECTORY + "/" + pub["bib"]["title"] + ".pdf")
            return DIRECTORY + "/" + pub["bib"]["title"].replace('/', '-') + ".pdf"
        

s = ScholarlyDownloader()