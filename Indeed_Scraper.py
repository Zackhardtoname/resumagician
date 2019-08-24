import re, locale
import json
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def get_soup(url):
    """
    Given the url of a page, this function returns the soup object.

    Parameters:
        url: the link to get soup object for

    Returns:
        soup: soup object
    """
    chrome_driver = "./bin/chromedriver"
    chrome_options = Options()
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options, executable_path=chrome_driver)

    driver.get(url)
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    driver.close()

    return soup


def grab_job_links(soup):
    """
    Grab all non-sponsored job posting links from a Indeed search result page using the given soup object

    Parameters:
        soup: the soup object corresponding to a search result page
                e.g. https://www.indeed.com/jobs?q=data+scientist&l=United+States&start=2

    Returns:
        urls: a python list of job posting urls

    """
    urls = []

    # Loop thru all the posting links
    for link in soup.find_all('div', {'class': 'title'}):
        # Since sponsored job postings are represented by "a target" instead of "a href", no need to worry here
        partial_url = link.a.get('href')
        # This is a partial url, we need to attach the prefix
        url = 'https://www.indeed.com' + partial_url
        # Make sure this is not a sponsored posting
        urls.append(url)

    return urls


def gen_base_url(query, location):
    """
    :param query: job title to query
    :param location: city to search in
    :return: a url to indeed website wrt the given query, num_pages and location
    """
    return 'https://www.indeed.com/jobs?q={}&l={}'.format(query, location)


def get_urls(base_url, num_pages):
    """
    Get all the job posting URLs resulted from a base_url.
    :param base_url:
    :param num_pages: num of pages needed
    :return：
        urls: a list of job posting URL's (when num_pages valid)
        max_pages: maximum number of pages allowed ((when num_pages invalid))
    """
    # We always need the first page
    soup = get_soup(base_url)
    urls = grab_job_links(soup)

    # Get the total number of postings found
    posting_count_string = soup.find(name='div', attrs={'id': "searchCount"}).get_text()
    print('BEFORE: posting_count_string: {}'.format(posting_count_string))  #\nPage 1 of 915 jobs
    posting_count_string = posting_count_string[posting_count_string.find('of') + 2:].strip()
    print('AFTER: posting_count_string: {}'.format(posting_count_string))   # 915 jobs
    posting_count_string = posting_count_string.split()[0]

    try:
        posting_count = int(posting_count_string)
    except ValueError:  # 33,915
        # posting_count = int(re.search('\d+', posting_count_string).group(0))
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        posting_count = locale.atoi(posting_count_string)
        print('posting_count: {}'.format(posting_count))
    # finally:
    #     posting_count = 330  # setting to 330 when unable to get the total
    #     pass

    # Limit number of pages to get
    max_pages = round(posting_count / 19) - 2   # 19 posts per page?
    if num_pages > max_pages:
        print('returning max_pages!!')
        return max_pages

        # Additional work is needed when more than 1 page is requested
    if num_pages >= 2:
        # Start loop from page 2 since page 1 has been dealt with above
        for i in range(2, num_pages + 1):
            num = (i - 1) * 10
            start_pg = '&start={}'.format(num)
            base_url = base_url + start_pg
            try:
                soup = get_soup(base_url)
                # We always combine the results back to the list
                urls += grab_job_links(soup)
            except:
                continue

    # Check to ensure the number of urls gotten is correct
    # assert len(urls) == num_pages * 10, "There are missing job links, check code!"

    return urls


def get_posting(url):
    """
    Get the text portion including both title and job description of the job posting from a given url

    Parameters:
        url: The job posting link

    Returns:
        title: the job title (if "data scientist" is in the title)
        text: the job posting content
    """
    # Get the url content as BS object
    soup = get_soup(url)

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.decompose()  # rip it out

    # The job title is held in the h3 tag
    title = soup.find(name='h3').getText().lower()
    posting = soup.find(name='div', attrs={'class': "jobsearch-JobComponent-description"}).get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in posting.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return title, text.lower()

    # if 'data scientist' in title:  # We'll proceed to grab the job posting text if the title is correct
    # All the text info is contained in the div element with the below class, extract the text.
    # posting = soup.find(name='div', attrs={'class': "jobsearch-JobComponent"}).get_text()
    # return title, posting.lower()
    # else:
    # return False

    # Get rid of numbers and symbols other than given
    # text = re.sub("[^a-zA-Z'+#&]", " ", text)
    # Convert to lower case and split to list and then set
    # text = text.lower().strip()

    # return text


def get_data(query, num_pages, location='San Francisco'):
    """
    Get all the job posting data and save in a json file using below structure:

    {<count>: {'title': ..., 'posting':..., 'url':...}...}

    The json file name has this format: ""<query>.json"

    Parameters:
        query: Indeed query keyword such as 'Data Scientist'
        num_pages: Number of search results needed
        location: location to search for

    Returns:
        postings_dict: Python dict including all posting data

    """
    # Convert the queried title to Indeed format
    query = '+'.join(query.lower().split())

    postings_dict = {}
    base_url = gen_base_url(query, location)
    urls = get_urls(base_url, num_pages)

    #  Continue only if the requested number of pages is valid (when invalid, a number is returned instead of list)
    if isinstance(urls, list):
        num_urls = len(urls)
        for i, url in enumerate(urls):
            try:
                title, text = get_posting(url)
                postings_dict[i] = {}
                postings_dict[i]['title'], postings_dict[i]['text'], postings_dict[i]['url'] = \
                    title, text, url
            except:
                continue

            percent = (i + 1) / num_urls
            # Print the progress the "end" arg keeps the message in the same line
            print("Progress: {:2.0f}%".format(100 * percent), end='\r')

        # Save the dict as json file
        file_name = query.replace('+', '_') + '.json'
        with open(file_name, 'w') as f:
            json.dump(postings_dict, f)

        print('All {} postings have been scraped and saved!'.format(num_urls))
        # return postings_dict
    else:
        print("Due to similar results, maximum number of pages is only {}. Please try again!".format(urls))


# If script is run directly, we'll take input from the user
if __name__ == "__main__":
    # queries = ["data scientist", "machine learning engineer", "data engineer"]

    # while True:
    #     query = input("Please enter the title to scrape data for: \n").lower()
    #     if query in queries:
    #         break
    #     else:
    #         print("Invalid title! Please try again.")
    #
    # while True:
    #     num_pages = input("Please enter the number of pages needed (integer only): \n")
    #     try:
    #         num_pages = int(num_pages)
    #         break
    #     except:
    #         print("Invalid number of pages! Please try again.")

    # get_data(query, num_pages, location='United States')

    get_data("software engineer", 20, location='United States')
