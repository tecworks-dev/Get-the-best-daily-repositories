import { NextResponse } from 'next/server';
import {
  checkFileExists,
  downloadImage,
  generateFileName,
  uploadToR2
} from '../../actions/upload-actions';

export async function POST(request: Request) {
  try {
    const contentType = request.headers.get('content-type') || '';
    let imageData: ArrayBuffer | Buffer;
    let filename: string;

    if (contentType.includes('multipart/form-data')) {
      const formData = await request.formData();
      const file = formData.get('file') as File;

      if (!file) {
        return NextResponse.json(
          { error: 'Please provide an image file' },
          { status: 400 }
        );
      }

      filename = await generateFileName(file.name.toLowerCase(), 'upload');
      imageData = await file.arrayBuffer();
    } else {
      const { imageUrl } = await request.json();

      if (!imageUrl) {
        return NextResponse.json(
          { error: 'Please provide an image URL' },
          { status: 400 }
        );
      }

      const url = new URL(imageUrl);
      const domain = url.hostname;
      filename = await generateFileName(domain, 'logo');

      const fileExists = await checkFileExists(filename);
      if (fileExists) {
        return NextResponse.json({
          success: true,
          url: `${process.env.R2_PUBLIC_URL}/${filename}`,
          cached: true
        });
      }

      imageData = await downloadImage(imageUrl);
    }

    const publicUrl = await uploadToR2(filename, imageData);

    return NextResponse.json({
      success: true,
      url: publicUrl,
      cached: false
    });

  } catch (error: any) {
    console.error('Upload error:', error);
    return NextResponse.json(
      { error: `Upload failed: ${error.message}` },
      { status: 500 }
    );
  }
}