from bs4 import BeautifulSoup

with open('./EurLex_data/eurlex_html_EN_NOT/21978A0914(01)_EN_NOT.html') as infile:
    html = infile.read()
soup = BeautifulSoup(html, 'html.parser')
elem = soup.findAll('div', {'class': 'texte'})
# print(elem)
print(elem[0].text)
