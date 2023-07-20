# import Module
from selenium import webdriver

# Create Chrome Object
driver = webdriver.Chrome('C:\Program Files\chromedriver_win32\chromedriver.exe')


def github_repo(user_name, pass_word,
				repository_name, descriptions=False,
				private=True, readme=False):
	
	# Open github login page
	driver.get('https://github.com/login')

	# Username
	username = driver.find_element_by_xpath('//*[@id="login_field"]')
	username.send_keys(user_name)

	# Password
	password = driver.find_element_by_xpath('//*[@id="password"]')
	password.send_keys(pass_word)

	# Click on signin button
	signin = driver.find_element_by_xpath(
		'//*[@id="Sign In"]/div[4]/form/input[14]')
	signin.click()

	# Create new repo.
	new_repo = driver.find_element_by_xpath('//*[@id="repos-container"]/h2/a')
	new_repo.click()

	# Enter Repo. name
	repositoryname = driver.find_element_by_xpath('//*[@id="repository_name"]')
	repositoryname.send_keys(repository_name)

	# Optional

	# Enter Description
	if descriptions:
		description = driver.find_element_by_xpath(
			'//*[@id="repository_description"]')
		description.send_keys(descriptions)

	# Private Mode
	if private:
		private = driver.find_element_by_xpath(
			'//*[@id="repository_visibility_private"]')
		private.click()

	# Create ReadMe File
	if readme:
		readme = driver.find_element_by_xpath(
			'//*[@id="repository_auto_init"]')
		readme.click()


github_repo("debugging-bug", "tigerlake@20019",
			"repo_1")

print("Repository created")

create_repo = driver.find_element_by_xpath(
	'//*[@id="new_repository"]/div[4]/button')

create_repo.click()
