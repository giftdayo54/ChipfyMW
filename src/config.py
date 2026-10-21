"""M-CHIP configuration: districts, CRS, thresholds, mandatory notices."""
PRIORITY_DISTRICTS = ["Chikwawa", "Nsanje", "Blantyre", "Thyolo", "Mulanje", "Phalombe",
                      "Zomba", "Chiradzulu", "Mwanza", "Neno", "Balaka", "Mangochi", "Machinga"]
GEOGRAPHIC_CRS = "EPSG:4326"
AREA_CRS = "EPSG:32736"          # WGS84 / UTM 36S: covers most of Malawi (area error small, documented)
ACCUMULATION_WINDOWS_DAYS = (1, 3, 5, 7, 14, 30)
SPI_SCALES_MONTHS = (1, 3, 6, 12)
MIN_SPI_YEARS = 30               # WMO guidance: >=30 years for a reliable fit
DISCLAIMER = ("M-CHIP is a decision-support tool. It does not replace official warnings "
              "issued by Malawi's Department of Disaster Management Affairs or the "
              "Department of Climate Change and Meteorological Services.")
