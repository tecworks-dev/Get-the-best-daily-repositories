package zmodem.util;

import org.apache.commons.io.input.RandomAccessFileInputStream;

import java.io.*;

public class CustomFile implements FileAdapter {
    File file = null;

    public CustomFile(File file) {
        super();
        this.file = file;
    }

    @Override
    public String getName() {
        return file.getName();
    }

    @Override
    public InputStream getInputStream() throws IOException {
        return RandomAccessFileInputStream.builder()
                .setCloseOnClose(true)
                .setRandomAccessFile(new RandomAccessFile(file, "r"))
                .setBufferSize(1024 * 8)
                .get();
    }

    @Override
    public OutputStream getOutputStream() throws IOException {
        return getOutputStream(false);
    }

    @Override
    public OutputStream getOutputStream(boolean append) throws IOException {
        return new BufferedOutputStream(new FileOutputStream(file, append));
    }

    @Override
    public FileAdapter getChild(String name) {
        if (name.equals(file.getName())) {
            return this;
        } else if (file.isDirectory()) {
            return new CustomFile(new File(file.getAbsolutePath(), name));
        }
        return null;

    }

    @Override
    public long length() {
        return file.length();
    }

    @Override
    public boolean isDirectory() {
        return file.isDirectory();
    }

    @Override
    public boolean exists() {
        return file.exists();
    }

}
