/**
 * Merkez adından şehir ismini çıkarır
 * @param centerName Merkez adı
 * @returns Şehir ismi
 */
export function extractCity(centerName: string): string {
  // Merkez adından şehir ismini çıkar
  const match = centerName.match(/(?:^|\s)-\s*([^-]+)$/);
  return match ? match[1].trim() : centerName;
} 