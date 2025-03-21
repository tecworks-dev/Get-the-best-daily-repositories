package ru.blackfan.bfscan.cli;

import java.io.File;
import java.net.URI;
import java.util.List;

public class CommandLineResult {
    
    public final List<File> inputFiles;
    public final URI apiUrl;
    public final String searchString;
    public final String mode;
    public final String renameMappingFile;
    public final boolean minifiedAnnotationsSupport;

    public CommandLineResult(List<File> inputFiles, URI apiUrl, String searchString, String mode, String renameMappingFile, boolean minifiedAnnotationsSupport) {
        this.inputFiles = inputFiles;
        this.apiUrl = apiUrl;
        this.searchString = searchString;
        this.mode = mode;
        this.renameMappingFile = renameMappingFile;
        this.minifiedAnnotationsSupport = minifiedAnnotationsSupport;
    }
    
}
