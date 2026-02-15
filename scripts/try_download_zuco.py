#!/usr/bin/env python3
"""
Attempt to download ZuCo data using alternative methods.

Tries multiple approaches:
1. Direct OSF API
2. OSF file listing
3. Alternative repositories
"""

import requests
import json
from pathlib import Path
from tqdm import tqdm
import time


def download_with_progress(url, output_path):
    """Download file with progress bar."""
    print(f"Attempting download from: {url}")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=output_path.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✓ Downloaded: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def try_osf_api_download():
    """Try using OSF API to find and download files."""
    
    print("=" * 80)
    print("Attempting OSF API Download")
    print("=" * 80)
    
    # OSF project ID for ZuCo
    project_id = "q3zws"
    api_url = f"https://api.osf.io/v2/nodes/{project_id}/files/"
    
    try:
        print(f"Fetching file list from OSF API...")
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if 'data' in data:
            print(f"Found {len(data['data'])} file entries")
            
            # Print available files
            for item in data['data'][:5]:  # Show first 5
                print(f"  - {item.get('attributes', {}).get('name', 'Unknown')}")
            
            # Try to get download links
            for item in data['data']:
                attrs = item.get('attributes', {})
                name = attrs.get('name', '')
                
                if '.mat' in name.lower() or 'task' in name.lower():
                    # Get download link
                    links = item.get('links', {})
                    download_link = links.get('download')
                    
                    if download_link:
                        print(f"\nFound downloadable file: {name}")
                        print(f"  Link: {download_link}")
                        
                        output_path = Path("data/raw/zuco") / name
                        if download_with_progress(download_link, output_path):
                            return True
            
        else:
            print("No data in API response")
            
    except Exception as e:
        print(f"OSF API failed: {e}")
    
    return False


def try_direct_links():
    """Try known direct download links."""
    
    print("\n" + "=" * 80)
    print("Trying Known Download Links")
    print("=" * 80)
    
    # These are common OSF download patterns
    # Note: These may not work without proper authentication
    links = [
        "https://osf.io/download/5c6f3a61c8ac0200199e7621/",  # Example
        "https://files.osf.io/v1/resources/q3zws/providers/osfstorage/",
    ]
    
    for link in links:
        print(f"\nTrying: {link}")
        try:
            response = requests.head(link, timeout=10, allow_redirects=True)
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                content_length = response.headers.get('content-length', 0)
                
                print(f"  Type: {content_type}")
                print(f"  Size: {int(content_length)/(1024**3):.2f} GB")
                
                # This looks promising, try download
                output_path = Path("data/raw/zuco/zuco_data.zip")
                if download_with_progress(link, output_path):
                    return True
                    
        except Exception as e:
            print(f"  Failed: {e}")
    
    return False


def main():
    print("ZuCo Data Download Attempt")
    print("=" * 80)
    print()
    
    # Try OSF API first
    if try_osf_api_download():
        print("\n✓ Download successful via OSF API!")
        return 0
    
    # Try direct links
    if try_direct_links():
        print("\n✓ Download successful via direct link!")
        return 0
    
    # All methods failed
    print("\n" + "=" * 80)
    print("Automated Download Not Available")
    print("=" * 80)
    print()
    print("OSF requires manual browser download due to:")
    print("  • Authentication requirements")
    print("  • Cookie-based sessions")
    print("  • Anti-bot protections")
    print()
    print("Please download manually from:")
    print("  https://osf.io/q3zws/")
    print()
    print("See HOW_TO_DOWNLOAD_ZUCO.md for detailed instructions")
    
    return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
