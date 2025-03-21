package ru.blackfan.bfscan.parsing.constants.file;

import jadx.core.xmlgen.ResContainer;
import java.io.StringReader;
import java.util.HashSet;
import java.util.Set;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.InputSource;
import ru.blackfan.bfscan.helpers.KeyValuePair;

public class ApkResources {
    private static final Logger logger = LoggerFactory.getLogger(ApkResources.class);

    public static Set<KeyValuePair> process(ResContainer resContainer, DocumentBuilderFactory factory) {
        Set<KeyValuePair> keyValuePairs = new HashSet<>();
        try {
            if (resContainer.getDataType() == ResContainer.DataType.RES_TABLE) {
                for (ResContainer subFile : resContainer.getSubFiles()) {
                    if (subFile.getDataType() == ResContainer.DataType.TEXT) {
                        String fileName = subFile.getFileName();
                        String xmlContent = subFile.getText().toString();
                        DocumentBuilder builder = factory.newDocumentBuilder();
                        Document doc = builder.parse(new InputSource(new StringReader(xmlContent)));
                        NodeList nodes = doc.getElementsByTagName("string");
                        for (int i = 0; i < nodes.getLength(); i++) {
                            Element element = (Element) nodes.item(i);
                            keyValuePairs.add(new KeyValuePair(fileName + "#" + element.getAttribute("name"), element.getTextContent()));
                        }
                    }
                }
            }
        } catch (Exception ex) {
            logger.error("Failed to process APK resource file", ex);
        }
        
        return keyValuePairs;
    }
}