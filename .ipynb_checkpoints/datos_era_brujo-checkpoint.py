import cdsapi

client = cdsapi.Client()

dataset = "reanalysis-era5-land-monthly-means"
request = {
    "product_type": "monthly_averaged_reanalysis",
    "variable": [
        "2m_temperature",
        "10m_u_component_of_wind",
        "total_precipitation"
    ],
    "year": [str(y) for y in range(1950, 2026)],  # años de 1950 a 2025
    "month": [
        "01", "02", "03", "04", "05", "06",
        "07", "08", "09", "10", "11", "12"
    ],
    "time": "00:00",
    "format": "netcdf",
    "area": [9.58, -83.81, 8.97, -83.21]  # N, W, S, E (Lat y Lon)
}

client.retrieve(dataset, request).download('era5_land_data_brujo.nc')
