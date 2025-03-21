package ru.blackfan.bfscan.helpers;

import io.swagger.v3.oas.models.PathItem;
import jadx.api.plugins.input.data.annotations.EncodedType;
import jadx.api.plugins.input.data.annotations.EncodedValue;
import jadx.core.dex.nodes.ClassNode;
import jadx.core.dex.nodes.RootNode;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.IOException;
import java.io.StringWriter;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.charset.Charset;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.zip.ZipFile;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.stream.StreamResult;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

public class Helpers {

    private static final Pattern VALID_URL_PATH_PATTERN = Pattern.compile("^[a-zA-Z0-9-._~!$&'()*+,;=:@/?#%{}]*$");
    
    public static boolean isValidRequestHeader(String header) {
        String regex = "^[A-Z].*-.*$";
        boolean matchesPattern = header.matches(regex);
        boolean containsAuthorization = header.toLowerCase().contains("authorization");
        return matchesPattern || containsAuthorization;
    }

    public static boolean isValidURI(String input) {
        try {
            URI uri = new URI(input);
            return uri.getScheme() != null;
        } catch (URISyntaxException e) {
            return false;
        }
    }

    public static boolean isValidUrlPath(String path) {
        if (path == null) {
            return false;
        }
        path = path.replaceAll("^\\.+", "");
        if (!path.startsWith("/")) {
            return false;
        }
        if (!VALID_URL_PATH_PATTERN.matcher(path).matches()) {
            return false;
        }
        try {
            URI uri = new URI("http", "example.com", path, null);
            return true;
        } catch (URISyntaxException e) {
            return false;
        }
    }
    
    public static ClassNode loadClass(RootNode root, String className) {
        ClassNode clsNode = root.resolveClass(className);
        if (clsNode != null) {
            clsNode.load();
            clsNode.reloadCode();
            return clsNode;
        }
        return null;
    }

    public static String getJvmAlias(RootNode root, String className) {
        ClassNode clsNode = loadClass(root, className);
        if (clsNode != null) {
            String alias = clsNode.getClassInfo().getAliasFullName();
            if (alias != null) {
                return "L" + alias.replace(".", "/") + ";";
            }
        }
        return null;
    }

    public static boolean isValidHttpMethod(String method) {
        for (PathItem.HttpMethod httpMethod : PathItem.HttpMethod.values()) {
            if (httpMethod.name().equalsIgnoreCase(method)) {
                return true;
            }
        }
        return false;
    }

    public static boolean isPureAscii(String v) {
        return Charset.forName("US-ASCII").newEncoder().canEncode(v);
    }

    public static String classSigToRawFullName(String clsSig) {
        if (clsSig != null && clsSig.startsWith("L") && clsSig.endsWith(";")) {
            clsSig = clsSig.substring(1, clsSig.length() - 1).replace("/", ".");
        }
        return clsSig;
    }

    public static String classSigToRawShortName(String clsSig) {
        if (clsSig == null) {
            return null;
        }
        if (clsSig.startsWith("L") && clsSig.endsWith(";")) {
            clsSig = clsSig.substring(1, clsSig.length() - 1).replace("/", ".");
        }
        return clsSig.substring(clsSig.lastIndexOf('.') + 1);
    }

    public static boolean isJVMSig(String sig) {
        if (sig == null) {
            return false;
        }
        if(sig.matches("^[a-zA-Z_\\$<][\\w\\$>]*\\(\\[?[a-zA-Z0-9\\$_/\\[\\];]*\\)([BCDFIJSVZ]|\\[?[L\\[a-zA-Z0-9\\$_/]+;)$")) {
            return true;
        }
        return false;
    }

    public static String stringWrapper(EncodedValue encodedValue) {
        if (encodedValue != null && encodedValue.getType() == EncodedType.ENCODED_STRING) {
            return (String) encodedValue.getValue();
        } else if (encodedValue == null) {
            return null;
        } else {
            return encodedValue.toString();
        }
    }

    public static String removeExtension(String filename) {
        int lastDotIndex = filename.lastIndexOf('.');
        if (lastDotIndex > 0) {
            return filename.substring(0, lastDotIndex);
        }
        return filename;
    }

    public static String getFileExtension(File file) {
        return getFileExtension(file.getName());
    }

    public static String getFileExtension(String name) {
        int lastDotIndex = name.lastIndexOf('.');
        if (lastDotIndex > 0) {
            return name.substring(lastDotIndex + 1);
        }
        return "";
    }

    public static ZipFile inputSteamToZipFile(InputStream is) throws IOException {
        File zipTemp = File.createTempFile("nested-", ".zip");
        zipTemp.deleteOnExit();

        FileOutputStream fos = new FileOutputStream(zipTemp);
        byte[] buffer = new byte[4096];
        int bytesRead;
        while ((bytesRead = is.read(buffer)) != -1) {
            fos.write(buffer, 0, bytesRead);
        }

        return new ZipFile(zipTemp);
    }

    public static String convertToXML(Map<String, Object> map, String rootElementName) throws ParserConfigurationException, Exception {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        DocumentBuilder builder = factory.newDocumentBuilder();
        Document document = builder.newDocument();

        Element root = document.createElement(rootElementName);
        document.appendChild(root);

        appendMapToElement(document, root, map);

        return transformDocumentToString(document);
    }

    private static void appendMapToElement(Document document, Element parent, Map<String, Object> map) {
        for (Map.Entry<String, Object> entry : map.entrySet()) {
            Element element = document.createElement(entry.getKey());
            if (entry.getValue() instanceof Map) {
                appendMapToElement(document, element, (Map<String, Object>) entry.getValue());
            } else if (entry.getValue() instanceof List) {
                appendListToElement(document, element, (List<Object>) entry.getValue());
            } else {
                element.appendChild(document.createTextNode(entry.getValue().toString()));
            }
            parent.appendChild(element);
        }
    }

    private static void appendListToElement(Document document, Element parent, List<Object> list) {
        for (Object item : list) {
            Element itemElement = document.createElement("item");
            if (item instanceof Map) {
                appendMapToElement(document, itemElement, (Map<String, Object>) item);
            } else {
                itemElement.appendChild(document.createTextNode(item.toString()));
            }
            parent.appendChild(itemElement);
        }
    }

    private static String transformDocumentToString(Document document) throws TransformerException {
        TransformerFactory transformerFactory = TransformerFactory.newInstance();
        Transformer transformer = transformerFactory.newTransformer();
        transformer.setOutputProperty(OutputKeys.INDENT, "yes");
        transformer.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "4");

        StringWriter writer = new StringWriter();
        transformer.transform(new DOMSource(document), new StreamResult(writer));
        return writer.toString();
    }
}
