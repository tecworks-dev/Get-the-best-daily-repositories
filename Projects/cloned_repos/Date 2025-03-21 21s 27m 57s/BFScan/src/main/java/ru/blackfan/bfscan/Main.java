package ru.blackfan.bfscan;

import jadx.api.args.UserRenamesMappingsMode;
import jadx.api.impl.NoOpCodeCache;
import jadx.api.impl.SimpleCodeWriter;
import jadx.api.JadxArgs;
import jadx.api.JadxDecompiler;
import jadx.plugins.mappings.RenameMappingsOptions;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.blackfan.bfscan.cli.CommandLineResult;
import ru.blackfan.bfscan.helpers.Helpers;
import ru.blackfan.bfscan.jadx.JadxBasePluginLoader;
import ru.blackfan.bfscan.parsing.constants.ConstantsProcessor;
import ru.blackfan.bfscan.parsing.httprequests.HTTPRequestProcessor;

public class Main {

    private static final Logger logger = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) {
        Options options = createOptions();
        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();

        try {
            final Path mappingPath;
            CommandLineResult cmdResult = parseCommandLine(parser, options, args);
            if (cmdResult.renameMappingFile != null) {
                mappingPath = Paths.get(cmdResult.renameMappingFile);
                if (!Files.exists(mappingPath) || !Files.isRegularFile(mappingPath) || !Files.isReadable(mappingPath)) {
                    throw new IllegalArgumentException("Mapping file is not available or not readable: " + cmdResult.renameMappingFile);
                }
            } else {
                mappingPath = null;
            }

            cmdResult.inputFiles.parallelStream().forEach(inputFile -> {
                try {
                    processInputFile(inputFile, mappingPath, cmdResult);
                } catch (Throwable e) {
                    logger.error("Error processing file " + inputFile.getName(), e);
                }
            });

        } catch (ParseException e) {
            handleError("Command line parsing error: " + e.getMessage(), e, formatter, options);
        } catch (Exception e) {
            handleError("Fatal error", e, formatter, options);
        }
    }

    private static void processInputFile(File inputFile, Path mappingPath, CommandLineResult cmdResult) {
        String reportUrlsPaths = inputFile.getAbsoluteFile().getParent() + File.separator + Helpers.removeExtension(inputFile.getName()) + "_urls.md";
        String reportSecrets = inputFile.getAbsoluteFile().getParent() + File.separator + Helpers.removeExtension(inputFile.getName()) + "_secrets.md";
        String reportMethods = inputFile.getAbsoluteFile().getParent() + File.separator + Helpers.removeExtension(inputFile.getName()) + "_methods.md";
        String reportOpenApi = inputFile.getAbsoluteFile().getParent() + File.separator + Helpers.removeExtension(inputFile.getName()) + "_openapi.yaml";
        String reportSearch = inputFile.getAbsoluteFile().getParent() + File.separator + Helpers.removeExtension(inputFile.getName()) + "_search.md";

        logger.info("Processing: " + inputFile.getName());

        try {
            PrintWriter writerSearch = cmdResult.searchString != null ? new PrintWriter(new FileWriter(reportSearch)) : null;

            JadxDecompiler jadx = initializeJadx(inputFile, mappingPath);

            if (List.of("a", "all", "s", "secrets", "secret").contains(cmdResult.mode)) {
                PrintWriter writerSecrets = new PrintWriter(new FileWriter(reportSecrets));
                PrintWriter writerUrlsPaths = new PrintWriter(new FileWriter(reportUrlsPaths));
                ConstantsProcessor constantsProcessor = new ConstantsProcessor(jadx, writerUrlsPaths, writerSecrets, writerSearch);
                constantsProcessor.processConstants(cmdResult.searchString);
                logger.info("URLs saved to: " + reportUrlsPaths);
                logger.info("Secrets saved to: " + reportSecrets);
            }

            if (List.of("a", "all", "h", "http").contains(cmdResult.mode)) {
                PrintWriter writerMethods = new PrintWriter(new FileWriter(reportMethods));
                PrintWriter writerOpenApi = new PrintWriter(new FileWriter(reportOpenApi));
                HTTPRequestProcessor processor = new HTTPRequestProcessor(jadx, cmdResult.apiUrl, cmdResult.minifiedAnnotationsSupport, writerMethods, writerOpenApi);
                processor.processHttpMethods();
                logger.info("Raw http methods saved to: " + reportMethods);
                logger.info("OpenAPI spec saved to: " + reportOpenApi);
            }

            if (cmdResult.searchString != null) {
                logger.info("Search result saved to: " + reportSearch);
            }

        } catch (IOException e) {
            throw new RuntimeException("Error writing output files for " + inputFile.getName(), e);
        } catch (Exception e) {
            throw new RuntimeException("Error processing file " + inputFile.getName(), e);
        }
    }

    private static Options createOptions() {
        Options options = new Options();

        Option urlOption = new Option("u", true, "API base url (http://localhost/api/)");
        urlOption.setRequired(false);
        urlOption.setArgName("url");

        Option verboseOption = new Option("v", true, "Log level (off, error, warn, info, debug, trace)");
        verboseOption.setRequired(false);
        verboseOption.setArgName("verbose");

        Option searchOption = new Option("s", true, "Search string");
        searchOption.setRequired(false);
        searchOption.setArgName("searchString");

        Option modeOption = new Option("m", true, "Mode ([a]ll, [s]ecrets, [h]ttp), default: all");
        modeOption.setRequired(false);
        modeOption.setArgName("mode");

        Option renameOption = new Option("r", true, "Rename mapping file");
        renameOption.setRequired(false);
        renameOption.setArgName("mappingFile");

        Option minifiedAnnotationsSupport = new Option("ma", true, "Minified or unknown annotations support (yes, no), default: yes");
        minifiedAnnotationsSupport.setRequired(false);
        minifiedAnnotationsSupport.setArgName("minifiedAnnotationsSupport");

        options.addOption(urlOption);
        options.addOption(verboseOption);
        options.addOption(searchOption);
        options.addOption(modeOption);
        options.addOption(renameOption);
        options.addOption(minifiedAnnotationsSupport);
        return options;
    }

    private static CommandLineResult parseCommandLine(CommandLineParser parser, Options options, String[] args)
            throws ParseException, MalformedURLException, URISyntaxException {
        CommandLine commandLine = parser.parse(options, args);
        List<String> cliArgs = commandLine.getArgList();
        if (cliArgs.isEmpty()) {
            throw new ParseException("Specify at least one input file (APK, XAPK, JAR, WAR or DEX)");
        }

        List<File> inputFiles = new ArrayList<>();
        for (String arg : cliArgs) {
            File file = new File(arg);
            validateInputFile(file);
            inputFiles.add(file);
        }

        URI apiUrl;
        if (commandLine.hasOption("u")) {
            apiUrl = new URL(commandLine.getOptionValue("u")).toURI();
        } else {
            apiUrl = new URL("http://localhost/").toURI();
        }

        if (commandLine.hasOption("v")) {
            System.setProperty("org.slf4j.simpleLogger.log.jadx", commandLine.getOptionValue("v"));
            System.setProperty("org.slf4j.simpleLogger.log.ru.blackfan", commandLine.getOptionValue("v"));
        }

        String searchString = null;
        if (commandLine.hasOption("s")) {
            searchString = commandLine.getOptionValue("s");
        }

        String mode;
        if (commandLine.hasOption("m")) {
            mode = commandLine.getOptionValue("m");
        } else {
            mode = "all";
        }

        String renameMappingPath = null;
        if (commandLine.hasOption("r")) {
            renameMappingPath = commandLine.getOptionValue("r");
        }

        boolean minifiedAnnotationsSupport = true;
        if (commandLine.hasOption("ma")) {
            switch (commandLine.getOptionValue("ma")) {
                case "no" -> minifiedAnnotationsSupport = false;
                case "yes" -> minifiedAnnotationsSupport = true;
                default -> throw new ParseException("Incorrect value in -ma option");
            }
        }

        return new CommandLineResult(inputFiles, apiUrl, searchString, mode, renameMappingPath, minifiedAnnotationsSupport);
    }

    private static JadxDecompiler initializeJadx(File inputFile, Path mappingPath) {
        JadxArgs jadxArgs = new JadxArgs();
        jadxArgs.setInputFile(inputFile);
        jadxArgs.setCodeCache(new NoOpCodeCache());
        jadxArgs.setReplaceConsts(false);
        jadxArgs.setDebugInfo(true);
        jadxArgs.setPluginLoader(new JadxBasePluginLoader());
        jadxArgs.setCodeWriterProvider(SimpleCodeWriter::new);
        jadxArgs.setMoveInnerClasses(false);
        jadxArgs.setShowInconsistentCode(true);
        jadxArgs.getPluginOptions().put("jadx.plugins.kotlin.metadata.fields", "true");
        if (mappingPath != null) {
            jadxArgs.setUserRenamesMappingsPath(mappingPath);
            jadxArgs.setUserRenamesMappingsMode(UserRenamesMappingsMode.READ);
            jadxArgs.getPluginOptions().put(RenameMappingsOptions.FORMAT_OPT, "AUTO");
            jadxArgs.getPluginOptions().put(RenameMappingsOptions.INVERT_OPT, "no");
        }
        JadxDecompiler jadx = new JadxDecompiler(jadxArgs);
        jadx.load();
        return jadx;
    }

    private static void validateInputFile(File file) throws IllegalArgumentException {
        if (!file.exists()) {
            throw new IllegalArgumentException("File does not exist: " + file.getPath());
        }
        if (!file.canRead()) {
            throw new IllegalArgumentException("Cannot read file: " + file.getPath());
        }
        String ext = Helpers.getFileExtension(file);
        if (!Arrays.asList("apk", "xapk", "jar", "war", "dex", "zip").contains(ext.toLowerCase())) {
            throw new IllegalArgumentException("Unsupported file type: " + ext);
        }
    }

    private static void handleError(String message, Throwable error, HelpFormatter formatter, Options options) {
        logger.error(message, error);
        formatter.setWidth(200);
        formatter.printHelp("java -jar bfscan.jar <jar_war_dex_apk_xapk_path> <...>", options, true);
        System.exit(1);
    }

}
