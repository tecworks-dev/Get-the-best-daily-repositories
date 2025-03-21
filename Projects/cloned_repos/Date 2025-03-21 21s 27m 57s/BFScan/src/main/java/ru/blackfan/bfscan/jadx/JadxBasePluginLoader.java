package ru.blackfan.bfscan.jadx;

import jadx.api.plugins.JadxPlugin;
import jadx.api.plugins.loader.JadxPluginLoader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class JadxBasePluginLoader implements JadxPluginLoader {
    
    @Override
    public List<JadxPlugin> load() {
        List<JadxPlugin> list = new ArrayList<>();
        list.add(new jadx.plugins.input.dex.DexInputPlugin());
        list.add(new jadx.plugins.input.java.JavaInputPlugin());
        list.add(new jadx.plugins.input.xapk.XapkInputPlugin());
        list.add(new jadx.plugins.kotlin.metadata.KotlinMetadataPlugin());
        list.add(new jadx.plugins.mappings.RenameMappingsPlugin());
        return list;
    }

    @Override
    public void close() throws IOException {
    }
    
}
