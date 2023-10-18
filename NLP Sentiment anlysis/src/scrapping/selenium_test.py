# import webdriver
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys 
import time
driver = webdriver.Chrome()
link = "https://www.youtube.com/"
driver.get(link)

search = driver.find_element(By.XPATH,'//*[@id="search-input"]')
search.click()
time.sleep(10)
# search.send_keys("Suraj")
# time.sleep(15)
# //*[@id="btfSubNavDesktopCustomerReviewsTab"]/div