#include <bits/stdc++.h>
using namespace std;

// crc_calculation.cpp

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <chrono>
using namespace std::chrono;

// Define the polynomial you want to use
#define CRC_POLYNOMIAL 0x93A409EB
#define CRC_INITIAL_VALUE 0xFFFFFFFF

typedef uint32_t crc_t;

// CRC table declaration
uint32_t crc_table[256];

// Function to generate the CRC lookup table
void generate_crc_table() {
    uint32_t polynomial = CRC_POLYNOMIAL;
    
    for (int i = 0; i < 256; i++) {
        uint32_t crc = (uint32_t)(i) << 24;
        for (int j = 0; j < 8; j++) {
            if (crc & 0x80000000) {
                crc = (crc << 1) ^ polynomial;
            } else {
                crc <<= 1;
            }
        }
        crc_table[i] = crc;
    }
}



// CRC initialization and finalization functions
crc_t crc_init(void) {
    return CRC_INITIAL_VALUE;
}

crc_t crc_finalize(crc_t crc) {
    // Return CRC as is, or XOR with 0xFFFFFFFF if required
    return crc;
}

// CRC update function using table lookup
crc_t crc_update(crc_t crc, const void *data, size_t data_len) {
    const uint8_t *d = (const uint8_t *)data;
    cout<<"Update Called"<<endl;

    while (data_len--) {
        // Reversing the bit order during CRC calculation
        uint8_t byte = *d++;
        // Reverse the byte to match the bit-by-bit MSB-to-LSB order
        byte = ((byte * 0x0802LU & 0x22110LU) | (byte * 0x8020LU & 0x88440LU)) * 0x10101LU >> 16;
        
        // Table lookup with reversed byte
        uint8_t tbl_idx = (uint8_t)((crc >> 24) ^ byte);
        crc = (crc << 8) ^ crc_table[tbl_idx];
    }

    return crc;
}

// Bit-by-bit CRC update function
uint32_t crc_update1(uint32_t crc, const void *data, size_t data_len) {
    const unsigned char *d = (const unsigned char *)data;

    unsigned int i;
    uint32_t bit;
    unsigned char c;
    cout<<"Update1 Called"<<endl;

    while (data_len--) {
        c = *d++;
        // Process bits from MSB to LSB
        for (i = 0x01; i &0xFF; i <<= 1) {
            bit = (crc & 0x80000000) ^ ((c & i) ? 0x80000000 : 0);
            crc <<= 1;
            if (bit) {
                crc ^= CRC_POLYNOMIAL;
            }
            crc &= 0xFFFFFFFF;
        }
    }

    return crc;
}


// Your original data packing function
uint32_t newCRC38(const uint32_t *ImgBufPtr, uint32_t TotPixelNum) {
    generate_crc_table();
    
    crc_t crc = crc_init();
    uint8_t data[19] = {0};

    uint32_t pixel0 = ImgBufPtr[0];
    uint32_t pixel1 = ImgBufPtr[1];
    uint32_t pixel2 = ImgBufPtr[2];
    uint32_t pixel3 = ImgBufPtr[3];

    // Pack pixel0
    data[0] = (pixel0 >> 24) & 0xFF;
    data[1] = (pixel0 >> 16) & 0xFF;
    data[2] = (pixel0 >> 8) & 0xFF;
    data[3] = pixel0 & 0xFF;

    // Pack pixel1
    data[4]  = (pixel1 << 6) & 0xFF;
    data[5]  = (pixel1 >> 2) & 0xFF;
    data[6]  = (pixel1 >> 10) & 0xFF;
    data[7]  = (pixel1 >> 18) & 0xFF;
    data[8]  = (pixel1 >> 26) & 0xFF;

    // Pack pixel2
    data[9]  = (pixel2 << 4) & 0xFF;
    data[10] = (pixel2 >> 4) & 0xFF;
    data[11] = (pixel2 >> 12) & 0xFF;
    data[12] = (pixel2 >> 20) & 0xFF;
    data[13] = (pixel2 >> 28) & 0xFF;

    // Pack pixel3
    data[14] = (pixel3 << 2) & 0xFF;
    data[15] = (pixel3 >> 6) & 0xFF;
    data[16] = (pixel3 >> 14) & 0xFF;
    data[17] = (pixel3 >> 22) & 0xFF;
    data[18] = (pixel3 >> 30) & 0xFF;

    // Update CRC with data
    auto start = high_resolution_clock::now();
    crc = crc_update(crc, data, 19);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << duration.count() << endl;
    // Finalize and return CRC
    crc = crc_finalize(crc);
    return crc;
}


int main() {
    uint32_t ImgBuf[4] = {0x12345678, 0x9ABCDEF0, 0x0FEDCBA9, 0x87654321};
    
    uint32_t crc = newCRC38(ImgBuf, 4); // Uses crc_update()
    
    //uint32_t crc1 = newCRC38_using_crc_update1(ImgBuf, 4); // Uses crc_update1()
    printf("CRC using crc_update:    %08X\n", crc);
    //printf("CRC using crc_update1:   %08X\n", crc1);

    return 0;
}
