"""
FIPS State Code to State Name Mapping

Standard FIPS codes for U.S. states, DC, and territories.
"""

FIPS_TO_STATE = {
    '01': 'Alabama',
    '02': 'Alaska',
    '04': 'Arizona',
    '05': 'Arkansas',
    '06': 'California',
    '08': 'Colorado',
    '09': 'Connecticut',
    '10': 'Delaware',
    '11': 'District of Columbia',
    '12': 'Florida',
    '13': 'Georgia',
    '15': 'Hawaii',
    '16': 'Idaho',
    '17': 'Illinois',
    '18': 'Indiana',
    '19': 'Iowa',
    '20': 'Kansas',
    '21': 'Kentucky',
    '22': 'Louisiana',
    '23': 'Maine',
    '24': 'Maryland',
    '25': 'Massachusetts',
    '26': 'Michigan',
    '27': 'Minnesota',
    '28': 'Mississippi',
    '29': 'Missouri',
    '30': 'Montana',
    '31': 'Nebraska',
    '32': 'Nevada',
    '33': 'New Hampshire',
    '34': 'New Jersey',
    '35': 'New Mexico',
    '36': 'New York',
    '37': 'North Carolina',
    '38': 'North Dakota',
    '39': 'Ohio',
    '40': 'Oklahoma',
    '41': 'Oregon',
    '42': 'Pennsylvania',
    '44': 'Rhode Island',
    '45': 'South Carolina',
    '46': 'South Dakota',
    '47': 'Tennessee',
    '48': 'Texas',
    '49': 'Utah',
    '50': 'Vermont',
    '51': 'Virginia',
    '53': 'Washington',
    '54': 'West Virginia',
    '55': 'Wisconsin',
    '56': 'Wyoming',
    # Territories (if applicable)
    '60': 'American Samoa',
    '66': 'Guam',
    '69': 'Northern Mariana Islands',
    '72': 'Puerto Rico',
    '78': 'U.S. Virgin Islands',
}

def get_state_name(fips_code):
    """Get state name from FIPS code."""
    return FIPS_TO_STATE.get(fips_code, f"Unknown (FIPS: {fips_code})")

if __name__ == "__main__":
    # Check the failed states
    failed_fips = ['03', '07', '14', '43', '52']
    
    print("Failed FIPS Codes and their corresponding states:")
    print("=" * 50)
    for fips in failed_fips:
        state = get_state_name(fips)
        print(f"  FIPS {fips}: {state}")
    
    print("\nNote: Some FIPS codes may not correspond to standard U.S. states.")
    print("They might be:")
    print("  - Territories")
    print("  - Invalid/outdated codes")
    print("  - Special administrative areas")

