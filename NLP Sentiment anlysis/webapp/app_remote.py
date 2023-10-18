from msdrive import OneDrive

# You need to get an access token from OneDrive API
# See https://docs.microsoft.com/en-us/onedrive/developer/rest-api/getting-started/?view=odsp-graph-online
access_token = "your_access_token_here"

# Create a OneDrive object with the access token
drive = OneDrive(access_token)

# Get the item id from the shared link
# See https://docs.microsoft.com/en-us/onedrive/developer/rest-api/api/drive_sharedwithme?view=odsp-graph-online
shared_link = "https://1drv.ms/u/s!AndgIBHrzel1gqc94ehPRp1Px2UUkg"
item_id = drive.get_item_id_from_shared_link(shared_link)

# Download the item to a local file
local_file = "downloaded_file.txt"
drive.download_item(item_id, local_file)

# Print a message to confirm the download
print(f"File downloaded from {shared_link} to {local_file}")
