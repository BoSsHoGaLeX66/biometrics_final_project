import asyncio
from bleak import BleakClient, BleakScanner

CHAR_UUID = "abcd1234-5678-1234-5678-abcdef123456"
DEVICE_NAME = "ESP32-Biometrics"

async def main():
    devices = await BleakScanner.discover()

    target = next((d for d in devices if d.name == DEVICE_NAME), None)
    if not target:
        print("ESP32 not found")
        return

    async with BleakClient(target.address) as client:
        print("Connected")

        while True:
            await client.write_gatt_char(CHAR_UUID, b"HIGH")
            await asyncio.sleep(1)

            await client.write_gatt_char(CHAR_UUID, b"LOW")
            await asyncio.sleep(1)

asyncio.run(main())