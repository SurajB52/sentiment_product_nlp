from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service as ChromeService

# Create a webdriver instance (make sure you have ChromeDriver installed and its path set)
driver = driver = uc.Chrome(headless=False,use_subprocess=False)

# Navigate to the page
url = "www.amazon"
driver.get(url)

# Locate the "Reviews" tab and click on it
reviews_tab = driver.find_element(By.ID, "btfSubNavDesktopCustomerReviewsTab")
reviews_tab.click()

# Wait for the reviews to load (you may need to adjust the wait time as needed)
wait = WebDriverWait(driver, 10)
reviews_element = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "tab-content")))

# Extract the reviews text
reviews_text = reviews_element.text

# Print or process the reviews text as needed
print(reviews_text)

# Close the webdriver
driver.quit()
