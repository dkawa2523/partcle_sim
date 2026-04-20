import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;

import java.io.BufferedWriter;
import java.io.PrintWriter;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ComsolParticleCaseExporter {
    private static Path mphPath;
    private static Path configPath;
    private static Path outDir;
    private static String caseName = "comsol_particle_case";
    private static String mode = "all";
    private static boolean exportMesh = true;
    private static boolean exportFields = true;
    private static String dataset = "dset1";
    private static String meshTag = "mesh1";
    private static int spatialDim = 2;
    private static int solnum = -1;
    private static String coordinateModelUnit = "m";
    private static double coordinateScaleMPerModelUnit = 1.0;
    private static String[] axisNames = new String[]{"x", "y"};
    private static double[] axisMin = new double[]{0.0, 0.0, 0.0};
    private static double[] axisMax = new double[]{1.0, 1.0, 1.0};
    private static int[] axisCount = new int[]{51, 51, 51};
    private static List<String> required = Arrays.asList("ux", "uy", "mu", "E_x", "E_y");
    private static Map<String, List<String>> expressions = new LinkedHashMap<>();

    public static void main(String[] args) throws Exception {
        loadOptions(args);
        loadConfig(configPath);
        Files.createDirectories(outDir);

        Model model = ModelUtil.load("particle_export_model", mphPath.toString());
        try {
            writeInventories(model);
            if (exportFields && ("all".equals(mode) || "fields".equals(mode))) {
                Map<String, String> selected = selectExpressions(model, probeCoordinates());
                writeExpressionInventory(outDir.resolve("expression_inventory.json"), selected);
                writeFieldSamples(outDir.resolve("field_samples.csv"), model, selected);
                writeManifest(outDir.resolve("export_manifest.json"), selected);
            } else {
                writeExpressionInventory(outDir.resolve("expression_inventory.json"), new LinkedHashMap<String, String>());
                writeManifest(outDir.resolve("export_manifest.json"), new LinkedHashMap<String, String>());
            }
        } finally {
            try {
                ModelUtil.disconnect();
            } catch (Throwable ignored) {
            }
        }
    }

    private static void loadOptions(String[] args) {
        Map<String, String> map = new LinkedHashMap<>();
        if (args != null) {
            for (int i = 0; i + 1 < args.length; i += 2) {
                map.put(args[i].replaceFirst("^-+", ""), args[i + 1]);
            }
        }
        mphPath = Paths.get(firstNonEmpty(map.get("mph"), System.getenv("COMSOL_PARTICLE_MPH"), "model.mph"));
        configPath = Paths.get(firstNonEmpty(map.get("config"), System.getenv("COMSOL_PARTICLE_CONFIG"), "external/comsol_particle_export/config/export_case.example.json"));
        outDir = Paths.get(firstNonEmpty(map.get("outdir"), System.getenv("COMSOL_PARTICLE_OUTDIR"), "_external_exports/comsol_particle_case"));
    }

    private static void loadConfig(Path path) throws Exception {
        String text = new String(Files.readAllBytes(path), StandardCharsets.UTF_8);
        caseName = jsonString(text, "case_name", caseName);
        mode = jsonString(text, "mode", mode);
        exportMesh = jsonBoolean(text, "export_mesh", exportMesh);
        exportFields = jsonBoolean(text, "export_fields", exportFields);
        dataset = jsonString(text, "dataset", dataset);
        meshTag = jsonString(text, "mesh_tag", meshTag);
        spatialDim = (int) jsonDouble(text, "spatial_dim", spatialDim);
        solnum = (int) jsonDouble(text, "solnum", solnum);
        coordinateModelUnit = jsonString(text, "coordinate_model_unit", coordinateModelUnit);
        coordinateScaleMPerModelUnit = jsonDouble(text, "coordinate_scale_m_per_model_unit", coordinateScaleMPerModelUnit);
        if (spatialDim < 1 || spatialDim > 3) {
            throw new IllegalArgumentException("spatial_dim must be 1, 2, or 3");
        }
        axisNames = jsonStringArray(text, "axis_names", defaultAxisNames(spatialDim));
        if (axisNames.length != spatialDim) {
            throw new IllegalArgumentException("axis_names length must match spatial_dim");
        }
        for (int d = 0; d < spatialDim; d++) {
            axisMin[d] = jsonDouble(text, "axis_" + d + "_min", axisMin[d]);
            axisMax[d] = jsonDouble(text, "axis_" + d + "_max", axisMax[d]);
            axisCount[d] = (int) jsonDouble(text, "axis_" + d + "_count", axisCount[d]);
            if (axisCount[d] < 2) {
                throw new IllegalArgumentException("axis_" + d + "_count must be at least 2");
            }
        }
        expressions = parseExpressionLists(text);
        expressions.remove("axis_names");
        if (expressions.containsKey("required")) {
            required = new ArrayList<>(expressions.get("required"));
            expressions.remove("required");
        }
    }

    private static void writeInventories(Model model) throws Exception {
        writeModelInventory(model, outDir.resolve("model_inventory.json"));
        writeMaterialInventory(model, outDir.resolve("material_inventory.json"));
        writeSelectionInventory(model, outDir.resolve("selection_inventory.json"));
        writePhysicsFeatureInventory(model, outDir.resolve("physics_feature_inventory.json"));
        writeParticleReleaseInventory(model, outDir.resolve("particle_release_inventory.json"));
        writeMethodList(outDir.resolve("model_methods.txt"), model);
        if (exportMesh && ("all".equals(mode) || "inventory".equals(mode) || "fields".equals(mode))) {
            exportMesh(model, outDir.resolve("mesh.mphtxt"));
        }
    }

    private static void writeModelInventory(Model model, Path out) throws Exception {
        String[] components = listTags(safeCall(model, "component"));
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            w.write("{\n");
            w.write("  \"source_kind\": \"external_comsol_particle_export_inventory\",\n");
            w.write("  \"case_name\": " + json(caseName) + ",\n");
            w.write("  \"mph_path\": " + json(mphPath.toString()) + ",\n");
            w.write("  \"mph_sha256\": " + json(sha256(mphPath)) + ",\n");
            w.write("  \"comsol_version\": " + json(stringOrEmpty(safeCall(model, "getComsolVersion"))) + ",\n");
            w.write("  \"title\": " + json(stringOrEmpty(safeCall(model, "title"))) + ",\n");
            w.write("  \"parameter_names\": " + jsonArray(listTagsLike(safeCall(model, "param"), "varnames")) + ",\n");
            w.write("  \"component_tags\": " + jsonArray(components) + ",\n");
            w.write("  \"study_tags\": " + jsonArray(listTags(safeCall(model, "study"))) + ",\n");
            w.write("  \"solver_tags\": " + jsonArray(listTags(safeCall(model, "sol"))) + ",\n");
            w.write("  \"dataset_tags\": " + jsonArray(listTags(safeCall(safeCall(model, "result"), "dataset"))) + ",\n");
            w.write("  \"components\": [\n");
            for (int i = 0; i < components.length; i++) {
                Object comp = safeCall(model, "component", components[i]);
                if (i > 0) {
                    w.write(",\n");
                }
                w.write("    {\n");
                w.write("      \"tag\": " + json(components[i]) + ",\n");
                w.write("      \"physics_tags\": " + jsonArray(listTags(safeCall(comp, "physics"))) + ",\n");
                w.write("      \"mesh_tags\": " + jsonArray(listTags(safeCall(comp, "mesh"))) + ",\n");
                w.write("      \"geometry_tags\": " + jsonArray(listTags(safeCall(comp, "geom"))) + ",\n");
                w.write("      \"selection_tags\": " + jsonArray(listTags(safeCall(comp, "selection"))) + "\n");
                w.write("    }");
            }
            w.write("\n  ]\n");
            w.write("}\n");
        }
    }

    private static void writeMaterialInventory(Model model, Path out) throws Exception {
        Object materials = safeCall(model, "material");
        String[] tags = listTags(materials);
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            w.write("{\n");
            w.write("  \"source_kind\": \"external_comsol_particle_export_material_inventory\",\n");
            w.write("  \"materials\": [\n");
            for (int i = 0; i < tags.length; i++) {
                Object material = safeCall(model, "material", tags[i]);
                Object selection = safeCall(material, "selection");
                if (i > 0) {
                    w.write(",\n");
                }
                w.write("    {\n");
                w.write("      \"tag\": " + json(tags[i]) + ",\n");
                w.write("      \"label\": " + json(stringOrEmpty(safeCall(material, "label"))) + ",\n");
                w.write("      \"name\": " + json(stringOrEmpty(safeCall(material, "name"))) + ",\n");
                w.write("      \"selection_entities\": " + jsonIntArray(selectionEntities(selection)) + "\n");
                w.write("    }");
            }
            w.write("\n  ]\n");
            w.write("}\n");
        }
    }

    private static void writeSelectionInventory(Model model, Path out) throws Exception {
        List<String> rows = new ArrayList<>();
        collectSelections(rows, model, "", "model");
        String[] components = listTags(safeCall(model, "component"));
        for (String component : components) {
            collectSelections(rows, safeCall(model, "component", component), component, "component");
        }
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            w.write("{\n");
            w.write("  \"source_kind\": \"external_comsol_particle_export_selection_inventory\",\n");
            w.write("  \"selections\": [\n");
            for (int i = 0; i < rows.size(); i++) {
                if (i > 0) {
                    w.write(",\n");
                }
                w.write(rows.get(i));
            }
            w.write("\n  ]\n");
            w.write("}\n");
        }
    }

    private static void collectSelections(List<String> rows, Object owner, String componentTag, String ownerKind) {
        String[] tags = listTags(safeCall(owner, "selection"));
        for (String tag : tags) {
            Object selection = safeCall(owner, "selection", tag);
            StringBuilder sb = new StringBuilder();
            sb.append("    {\n");
            sb.append("      \"owner_kind\": ").append(json(ownerKind)).append(",\n");
            sb.append("      \"component_tag\": ").append(json(componentTag)).append(",\n");
            sb.append("      \"tag\": ").append(json(tag)).append(",\n");
            sb.append("      \"label\": ").append(json(stringOrEmpty(safeCall(selection, "label")))).append(",\n");
            sb.append("      \"name\": ").append(json(stringOrEmpty(safeCall(selection, "name")))).append(",\n");
            sb.append("      \"entities\": ").append(jsonIntArray(selectionEntities(selection))).append("\n");
            sb.append("    }");
            rows.add(sb.toString());
        }
    }

    private static void writePhysicsFeatureInventory(Model model, Path out) throws Exception {
        String[] components = listTags(safeCall(model, "component"));
        List<String> rows = new ArrayList<>();
        for (String component : components) {
            Object comp = safeCall(model, "component", component);
            String[] physicsTags = listTags(safeCall(comp, "physics"));
            for (String physicsTag : physicsTags) {
                Object physics = safeCall(comp, "physics", physicsTag);
                String physicsLabel = stringOrEmpty(safeCall(physics, "label"));
                String physicsType = firstNonEmpty(
                    stringOrEmpty(safeCall(physics, "getType")),
                    stringOrEmpty(safeCall(physics, "type"))
                );
                String[] featureTags = listTags(safeCall(physics, "feature"));
                for (String featureTag : featureTags) {
                    Object feature = safeCall(physics, "feature", featureTag);
                    String label = stringOrEmpty(safeCall(feature, "label"));
                    String type = firstNonEmpty(
                        stringOrEmpty(safeCall(feature, "getType")),
                        stringOrEmpty(safeCall(feature, "type"))
                    );
                    String forceKind = classifyForceKind(physicsTag, physicsLabel, physicsType, featureTag, label, type);
                    StringBuilder sb = new StringBuilder();
                    sb.append("    {\n");
                    sb.append("      \"component_tag\": ").append(json(component)).append(",\n");
                    sb.append("      \"physics_tag\": ").append(json(physicsTag)).append(",\n");
                    sb.append("      \"physics_label\": ").append(json(physicsLabel)).append(",\n");
                    sb.append("      \"physics_type\": ").append(json(physicsType)).append(",\n");
                    sb.append("      \"feature_tag\": ").append(json(featureTag)).append(",\n");
                    sb.append("      \"label\": ").append(json(label)).append(",\n");
                    sb.append("      \"type\": ").append(json(type)).append(",\n");
                    sb.append("      \"force_kind\": ").append(json(forceKind)).append(",\n");
                    sb.append("      \"selection_entities\": ").append(jsonIntArray(selectionEntities(safeCall(feature, "selection")))).append(",\n");
                    sb.append("      \"known_settings\": ").append(featureKnownSettingsJson(feature)).append("\n");
                    sb.append("    }");
                    rows.add(sb.toString());
                }
            }
        }
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            w.write("{\n");
            w.write("  \"source_kind\": \"external_comsol_particle_export_physics_feature_inventory\",\n");
            w.write("  \"features\": [\n");
            for (int i = 0; i < rows.size(); i++) {
                if (i > 0) {
                    w.write(",\n");
                }
                w.write(rows.get(i));
            }
            w.write("\n  ]\n");
            w.write("}\n");
        }
    }

    private static void writeParticleReleaseInventory(Model model, Path out) throws Exception {
        String[] components = listTags(safeCall(model, "component"));
        List<String> rows = new ArrayList<>();
        for (String component : components) {
            Object comp = safeCall(model, "component", component);
            String[] physicsTags = listTags(safeCall(comp, "physics"));
            for (String physicsTag : physicsTags) {
                Object physics = safeCall(comp, "physics", physicsTag);
                String physicsLabel = stringOrEmpty(safeCall(physics, "label"));
                String physicsType = firstNonEmpty(
                    stringOrEmpty(safeCall(physics, "getType")),
                    stringOrEmpty(safeCall(physics, "type"))
                );
                String[] featureTags = listTags(safeCall(physics, "feature"));
                for (String featureTag : featureTags) {
                    Object feature = safeCall(physics, "feature", featureTag);
                    String label = stringOrEmpty(safeCall(feature, "label"));
                    String type = firstNonEmpty(
                        stringOrEmpty(safeCall(feature, "getType")),
                        stringOrEmpty(safeCall(feature, "type"))
                    );
                    String kind = classifyParticleReleaseKind(physicsTag, physicsLabel, physicsType, featureTag, label, type);
                    if ("other".equals(kind)) {
                        continue;
                    }
                    StringBuilder sb = new StringBuilder();
                    sb.append("    {\n");
                    sb.append("      \"component_tag\": ").append(json(component)).append(",\n");
                    sb.append("      \"physics_tag\": ").append(json(physicsTag)).append(",\n");
                    sb.append("      \"physics_label\": ").append(json(physicsLabel)).append(",\n");
                    sb.append("      \"physics_type\": ").append(json(physicsType)).append(",\n");
                    sb.append("      \"feature_tag\": ").append(json(featureTag)).append(",\n");
                    sb.append("      \"label\": ").append(json(label)).append(",\n");
                    sb.append("      \"type\": ").append(json(type)).append(",\n");
                    sb.append("      \"release_kind\": ").append(json(kind)).append(",\n");
                    sb.append("      \"selection_entities\": ").append(jsonIntArray(selectionEntities(safeCall(feature, "selection")))).append(",\n");
                    sb.append("      \"property_names\": ").append(jsonArray(featurePropertyNames(feature))).append(",\n");
                    sb.append("      \"known_settings\": ").append(particleReleaseSettingsJson(feature)).append("\n");
                    sb.append("    }");
                    rows.add(sb.toString());
                }
            }
        }
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            w.write("{\n");
            w.write("  \"source_kind\": \"external_comsol_particle_export_release_inventory\",\n");
            w.write("  \"features\": [\n");
            for (int i = 0; i < rows.size(); i++) {
                if (i > 0) {
                    w.write(",\n");
                }
                w.write(rows.get(i));
            }
            w.write("\n  ]\n");
            w.write("}\n");
        }
    }

    private static String classifyParticleReleaseKind(String... values) {
        String text = String.join(" ", values).toLowerCase(Locale.ROOT);
        if (containsAny(text, "grid") && containsAny(text, "release", "inlet", "inject", "source")) {
            return "release_grid";
        }
        if (containsAny(text, "release", "inlet", "inject", "source", "initial position", "initial coordinates")) {
            return "release";
        }
        if (containsAny(text, "initial velocity", "velocity direction", "velocity magnitude")) {
            return "initial_velocity";
        }
        if (containsAny(text, "particle properties", "particle property", "diameter", "mass", "density")) {
            return "particle_properties";
        }
        return "other";
    }

    private static String classifyForceKind(String... values) {
        String text = String.join(" ", values).toLowerCase(Locale.ROOT);
        if (containsAny(text, "thermophor", "thermophoretic")) {
            return "thermophoresis";
        }
        if (containsAny(text, "dielectrophor", "dielectrophoretic", "dep")) {
            return "dielectrophoresis";
        }
        if (containsAny(text, "saffman", "lift")) {
            return "lift";
        }
        if (containsAny(text, "gravity", "gravit")) {
            return "gravity";
        }
        if (containsAny(text, "brownian", "langevin")) {
            return "brownian";
        }
        if (containsAny(text, "drag", "stokes", "epstein")) {
            return "drag";
        }
        if (containsAny(text, "electric", "electrostatic", "coulomb")) {
            return "electric";
        }
        if (containsAny(text, "magnetic", "lorentz")) {
            return "magnetic";
        }
        return "other";
    }

    private static boolean containsAny(String text, String... needles) {
        for (String needle : needles) {
            if (text.contains(needle)) {
                return true;
            }
        }
        return false;
    }

    private static String featureKnownSettingsJson(Object feature) {
        String[] keys = new String[]{
            "F", "Fx", "Fy", "Fz",
            "g", "g_const", "gvec",
            "T", "rho", "mu", "eta",
            "k", "kg", "kp", "Cs", "Cm", "Ct",
            "E", "Ex", "Ey", "Ez", "V",
            "epsilonr", "epsilonrp", "sigma", "sigmap", "freq",
            "u", "v", "w", "U", "walllift"
        };
        List<String> items = new ArrayList<>();
        for (String key : keys) {
            String value = featureSetting(feature, key);
            if (value == null || value.trim().isEmpty()) {
                continue;
            }
            items.add(json(key) + ": " + json(value));
        }
        return "{" + String.join(", ", items) + "}";
    }

    private static String particleReleaseSettingsJson(Object feature) {
        String[] keys = new String[]{
            "N", "n", "Np", "nump", "number", "nParticles", "npart",
            "t", "t0", "t1", "tlist", "times", "release_times", "releaseTime", "trelease", "tRelease",
            "period", "frequency", "f", "phase", "pulse", "duration", "tstart", "tend", "dt",
            "grid", "gridtype", "gridType", "Nx", "Ny", "Nz", "Nr", "Nz_grid", "n0", "n1", "n2",
            "x0", "y0", "z0", "r0", "x", "y", "z", "r", "coord", "coords",
            "vx0", "vy0", "vz0", "vr0", "v0", "v", "speed", "direction", "normal",
            "diameter", "dp", "radius", "rp", "rho", "density", "mass", "mp",
            "charge", "q", "material", "selection", "distrib", "distribution"
        };
        List<String> items = new ArrayList<>();
        for (String key : keys) {
            String value = featureSetting(feature, key);
            if (value == null || value.trim().isEmpty()) {
                continue;
            }
            items.add(json(key) + ": " + json(value));
        }
        return "{" + String.join(", ", items) + "}";
    }

    private static String[] featurePropertyNames(Object feature) {
        for (String method : new String[]{"properties", "getProperties", "propertyNames", "getPropertyNames"}) {
            Object value = safeCall(feature, method);
            if (value instanceof String[]) {
                return (String[]) value;
            }
        }
        return new String[0];
    }

    private static String featureSetting(Object feature, String key) {
        for (String method : new String[]{"getString", "get"}) {
            try {
                Object value = call(feature, method, key);
                if (value != null) {
                    return String.valueOf(value);
                }
            } catch (Throwable ignored) {
            }
        }
        return "";
    }

    private static void exportMesh(Model model, Path out) {
        List<String> errors = new ArrayList<>();
        try {
            writeMphtxtFromMeshSequence(call(model, "mesh", meshTag), out);
            return;
        } catch (Throwable t) {
            errors.add(t.toString());
        }
        for (String component : listTags(safeCall(model, "component"))) {
            try {
                Object comp = call(model, "component", component);
                writeMphtxtFromMeshSequence(call(comp, "mesh", meshTag), out);
                return;
            } catch (Throwable t) {
                errors.add(t.toString());
            }
        }
        throw new RuntimeException("Could not export mesh.mphtxt. Errors: " + errors);
    }

    private static void writeMphtxtFromMeshSequence(Object mesh, Path out) throws Exception {
        double[][] vertices = normalizeVertices((double[][]) call(mesh, "getVertex"));
        String[] types = (String[]) call(mesh, "getTypes");
        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(out, StandardCharsets.UTF_8))) {
            int sdim = vertices.length == 0 ? 0 : vertices[0].length;
            writer.println(sdim + " # sdim");
            writer.println(vertices.length + " # number of mesh vertices");
            writer.println("# Mesh vertex coordinates");
            for (double[] vertex : vertices) {
                for (int d = 0; d < sdim; d++) {
                    if (d > 0) {
                        writer.print(" ");
                    }
                    writer.print(String.format(Locale.US, "%.17g", vertex[d]));
                }
                writer.println();
            }
            writer.println(types.length + " # number of element types");
            for (int typeIndex = 0; typeIndex < types.length; typeIndex++) {
                String type = types[typeIndex];
                int[] entity = (int[]) call(mesh, "getElemEntity", type);
                int[][] elems = normalizeElements((int[][]) call(mesh, "getElem", type), entity.length);
                int nvp = elems.length == 0 ? 0 : elems[0].length;
                writer.println(typeIndex + " " + type + " # type name");
                writer.println(nvp + " # number of vertices per element");
                writer.println(elems.length + " # number of elements");
                writer.println("# Elements");
                for (int[] elem : elems) {
                    for (int j = 0; j < nvp; j++) {
                        if (j > 0) {
                            writer.print(" ");
                        }
                        writer.print(elem[j]);
                    }
                    writer.println();
                }
                writer.println(entity.length + " # number of geometric entity indices");
                writer.println("# Geometric entity indices");
                for (int value : entity) {
                    writer.println(value);
                }
            }
        }
    }

    private static Map<String, String> selectExpressions(Model model, double[][] probeCoords) {
        Map<String, String> selected = new LinkedHashMap<>();
        Map<String, String> failures = new LinkedHashMap<>();
        for (Map.Entry<String, List<String>> entry : expressions.entrySet()) {
            String key = entry.getKey();
            List<String> fail = new ArrayList<>();
            for (String expr : entry.getValue()) {
                String tag = "inv_" + sanitize(key);
                try {
                    Object interp = createInterp(model, tag, expr);
                    double value = evalFirstFinite(interp, probeCoords);
                    if (Double.isFinite(value)) {
                        selected.put(key, expr);
                        failures.put(key, "");
                        break;
                    }
                    fail.add(expr + ": non-finite");
                } catch (Throwable t) {
                    fail.add(expr + ": " + t.getClass().getSimpleName() + ": " + t.getMessage());
                } finally {
                    removeNumerical(model, tag);
                }
            }
            if (!selected.containsKey(key)) {
                failures.put(key, String.join("; ", fail));
            }
        }
        for (String key : required) {
            if (!selected.containsKey(key)) {
                throw new RuntimeException("Required COMSOL expression not found for " + key + ": " + failures.get(key));
            }
        }
        return selected;
    }

    private static Object createInterp(Model model, String tag, String expr) {
        Object result = call(model, "result");
        Object numerical = call(result, "numerical");
        try {
            call(numerical, "remove", tag);
        } catch (Throwable ignored) {
        }
        call(numerical, "create", tag, "Interp");
        Object interp = call(result, "numerical", tag);
        call(interp, "set", "data", dataset);
        if (solnum > 0) {
            call(interp, "set", "solnum", new int[]{solnum});
        }
        call(interp, "set", "expr", new String[]{expr});
        return interp;
    }

    private static void removeNumerical(Model model, String tag) {
        try {
            call(call(call(model, "result"), "numerical"), "remove", tag);
        } catch (Throwable ignored) {
        }
    }

    private static void writeFieldSamples(Path out, Model model, Map<String, String> selected) throws Exception {
        Map<String, Object> features = new LinkedHashMap<>();
        for (Map.Entry<String, String> entry : selected.entrySet()) {
            features.put(entry.getKey(), createInterp(model, "grid_" + sanitize(entry.getKey()), entry.getValue()));
        }
        double[][] coords = gridCoordinates();
        Map<String, double[]> valuesByKey = new LinkedHashMap<>();
        for (String key : selected.keySet()) {
            valuesByKey.put(key, evalMany(features.get(key), coords));
        }
        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(out, StandardCharsets.UTF_8))) {
            for (int d = 0; d < spatialDim; d++) {
                if (d > 0) {
                    writer.print(",");
                }
                writer.print(axisNames[d]);
            }
            writer.print(",valid_mask");
            for (String key : selected.keySet()) {
                writer.print(",");
                writer.print(key);
            }
            writer.println();
            for (int i = 0; i < coords[0].length; i++) {
                boolean valid = true;
                for (String key : selected.keySet()) {
                    if (required.contains(key) && !Double.isFinite(valuesByKey.get(key)[i])) {
                        valid = false;
                    }
                }
                for (int d = 0; d < spatialDim; d++) {
                    if (d > 0) {
                        writer.print(",");
                    }
                    writer.print(String.format(Locale.US, "%.17g", coords[d][i]));
                }
                writer.print(",");
                writer.print(valid ? "1" : "0");
                for (String key : selected.keySet()) {
                    double value = valuesByKey.get(key)[i];
                    writer.print(",");
                    writer.print(Double.isFinite(value) ? String.format(Locale.US, "%.17g", value) : "NaN");
                }
                writer.println();
            }
        } finally {
            for (String key : selected.keySet()) {
                removeNumerical(model, "grid_" + sanitize(key));
            }
        }
    }

    private static void writeExpressionInventory(Path out, Map<String, String> selected) throws Exception {
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            w.write("{\n");
            w.write("  \"source_kind\": \"external_comsol_particle_export_expression_inventory\",\n");
            w.write("  \"dataset\": " + json(dataset) + ",\n");
            w.write("  \"required\": " + jsonArray(required.toArray(new String[0])) + ",\n");
            w.write("  \"selected\": {\n");
            int i = 0;
            for (String key : expressions.keySet()) {
                if (i++ > 0) {
                    w.write(",\n");
                }
                w.write("    " + json(key) + ": {");
                w.write("\"expression\": " + json(selected.containsKey(key) ? selected.get(key) : "") + ", ");
                w.write("\"dataset\": " + json(selected.containsKey(key) ? dataset : "") + ", ");
                w.write("\"available\": " + selected.containsKey(key));
                w.write("}");
            }
            w.write("\n  }\n");
            w.write("}\n");
        }
    }

    private static void writeManifest(Path out, Map<String, String> selected) throws Exception {
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            w.write("{\n");
            w.write("  \"source_kind\": \"external_comsol_particle_export\",\n");
            w.write("  \"case_name\": " + json(caseName) + ",\n");
            w.write("  \"mph_path\": " + json(mphPath.toString()) + ",\n");
            w.write("  \"mph_sha256\": " + json(sha256(mphPath)) + ",\n");
            w.write("  \"dataset\": " + json(dataset) + ",\n");
            w.write("  \"mesh_tag\": " + json(meshTag) + ",\n");
            w.write("  \"spatial_dim\": " + spatialDim + ",\n");
            w.write("  \"axis_names\": " + jsonArray(axisNames) + ",\n");
            w.write("  \"coordinate_model_unit\": " + json(coordinateModelUnit) + ",\n");
            w.write("  \"coordinate_scale_m_per_model_unit\": " + jsonNumber(coordinateScaleMPerModelUnit) + ",\n");
            w.write("  \"grid_shape\": " + jsonIntArray(Arrays.copyOf(axisCount, spatialDim)) + ",\n");
            w.write("  \"expression_mapping\": {\n");
            int i = 0;
            for (Map.Entry<String, String> entry : selected.entrySet()) {
                if (i++ > 0) {
                    w.write(",\n");
                }
                w.write("    " + json(entry.getKey()) + ": " + json(entry.getValue()));
            }
            w.write("\n  }\n");
            w.write("}\n");
        }
    }

    private static double evalFirstFinite(Object interp, double[][] coords) {
        double[] values = evalMany(interp, coords);
        for (double value : values) {
            if (Double.isFinite(value)) {
                return value;
            }
        }
        return Double.NaN;
    }

    private static double[] evalMany(Object interp, double[][] coords) {
        call(interp, "setInterpolationCoordinates", coords);
        Object data = call(interp, "getData");
        return firstVector(data, coords[0].length);
    }

    private static double[][] gridCoordinates() {
        int n = 1;
        for (int d = 0; d < spatialDim; d++) {
            n *= axisCount[d];
        }
        double[][] coords = new double[spatialDim][n];
        int idx = 0;
        if (spatialDim == 1) {
            for (double x : linspace(axisMin[0], axisMax[0], axisCount[0])) {
                coords[0][idx++] = x;
            }
            return coords;
        }
        if (spatialDim == 2) {
            double[] a0 = linspace(axisMin[0], axisMax[0], axisCount[0]);
            double[] a1 = linspace(axisMin[1], axisMax[1], axisCount[1]);
            for (double x : a0) {
                for (double y : a1) {
                    coords[0][idx] = x;
                    coords[1][idx] = y;
                    idx++;
                }
            }
            return coords;
        }
        double[] a0 = linspace(axisMin[0], axisMax[0], axisCount[0]);
        double[] a1 = linspace(axisMin[1], axisMax[1], axisCount[1]);
        double[] a2 = linspace(axisMin[2], axisMax[2], axisCount[2]);
        for (double x : a0) {
            for (double y : a1) {
                for (double z : a2) {
                    coords[0][idx] = x;
                    coords[1][idx] = y;
                    coords[2][idx] = z;
                    idx++;
                }
            }
        }
        return coords;
    }

    private static double[][] probeCoordinates() {
        int n = Math.max(3, spatialDim * 3);
        double[][] coords = new double[spatialDim][n];
        for (int i = 0; i < n; i++) {
            double f = (i + 1.0) / (n + 1.0);
            for (int d = 0; d < spatialDim; d++) {
                coords[d][i] = axisMin[d] + f * (axisMax[d] - axisMin[d]);
            }
        }
        return coords;
    }

    private static double[] linspace(double min, double max, int count) {
        double[] out = new double[count];
        double step = (max - min) / (count - 1);
        for (int i = 0; i < count; i++) {
            out[i] = min + step * i;
        }
        return out;
    }

    private static double[] firstVector(Object data, int expected) {
        if (data instanceof double[][][]) {
            double[][][] a = (double[][][]) data;
            return a.length == 0 || a[0].length == 0 ? filledNaN(expected) : padded(a[0][0], expected);
        }
        if (data instanceof double[][]) {
            double[][] a = (double[][]) data;
            return a.length == 0 ? filledNaN(expected) : padded(a[0], expected);
        }
        if (data instanceof double[]) {
            return padded((double[]) data, expected);
        }
        if (data instanceof Double) {
            double[] out = filledNaN(expected);
            out[0] = (Double) data;
            return out;
        }
        return filledNaN(expected);
    }

    private static double[] padded(double[] values, int expected) {
        if (values.length == expected) {
            return values;
        }
        double[] out = filledNaN(expected);
        int n = Math.min(values.length, expected);
        for (int i = 0; i < n; i++) {
            out[i] = values[i];
        }
        return out;
    }

    private static double[] filledNaN(int n) {
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = Double.NaN;
        }
        return out;
    }

    private static double[][] normalizeVertices(double[][] raw) {
        if (raw.length == 0) {
            return raw;
        }
        if (raw.length <= 3 && raw[0].length > raw.length) {
            int sdim = raw.length;
            int n = raw[0].length;
            double[][] out = new double[n][sdim];
            for (int d = 0; d < sdim; d++) {
                for (int i = 0; i < n; i++) {
                    out[i][d] = raw[d][i];
                }
            }
            return out;
        }
        return raw;
    }

    private static int[][] normalizeElements(int[][] raw, int elementCount) {
        if (raw.length == 0 || elementCount == raw.length) {
            return raw;
        }
        if (raw[0].length == elementCount) {
            int nvp = raw.length;
            int[][] out = new int[elementCount][nvp];
            for (int j = 0; j < nvp; j++) {
                for (int i = 0; i < elementCount; i++) {
                    out[i][j] = raw[j][i];
                }
            }
            return out;
        }
        return raw;
    }

    private static Object call(Object target, String name, Object... args) {
        if (target == null) {
            throw new RuntimeException("Cannot call " + name + " on null target");
        }
        Throwable last = null;
        for (Method method : target.getClass().getMethods()) {
            if (!method.getName().equals(name) || method.getParameterCount() != args.length) {
                continue;
            }
            try {
                return method.invoke(target, args);
            } catch (InvocationTargetException t) {
                Throwable cause = t.getCause() == null ? t : t.getCause();
                throw new RuntimeException("Method " + name + " threw: " + cause.getMessage(), cause);
            } catch (Throwable t) {
                last = t;
            }
        }
        throw new RuntimeException("No callable method " + name + " with " + args.length + " args on " + target.getClass(), last);
    }

    private static Object safeCall(Object target, String name, Object... args) {
        try {
            return call(target, name, args);
        } catch (Throwable t) {
            return null;
        }
    }

    private static void writeMethodList(Path path, Object target) {
        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(path, StandardCharsets.UTF_8))) {
            if (target == null) {
                return;
            }
            writer.println(target.getClass().getName());
            for (Method method : target.getClass().getMethods()) {
                Class[] params = method.getParameterTypes();
                List<String> names = new ArrayList<>();
                for (Class param : params) {
                    names.add(param.getName());
                }
                writer.println(method.getName() + "(" + String.join(",", names) + ") -> " + method.getReturnType().getName());
            }
        } catch (Throwable ignored) {
        }
    }

    private static String[] listTags(Object list) {
        return listTagsLike(list, "tags");
    }

    private static String[] listTagsLike(Object list, String method) {
        Object raw = safeCall(list, method);
        if (raw instanceof String[]) {
            return (String[]) raw;
        }
        return new String[0];
    }

    private static int[] selectionEntities(Object selection) {
        if (selection == null) {
            return new int[0];
        }
        for (String method : new String[]{"entities", "inputEntities"}) {
            int[] out = normalizeIntArray(safeCall(selection, method));
            if (out != null) {
                return out;
            }
        }
        return new int[0];
    }

    private static int[] normalizeIntArray(Object value) {
        if (value instanceof int[]) {
            return (int[]) value;
        }
        if (value instanceof Integer[]) {
            Integer[] raw = (Integer[]) value;
            int[] out = new int[raw.length];
            for (int i = 0; i < raw.length; i++) {
                out[i] = raw[i] == null ? 0 : raw[i];
            }
            return out;
        }
        return null;
    }

    private static String stringOrEmpty(Object value) {
        return value == null ? "" : String.valueOf(value);
    }

    private static Map<String, List<String>> parseExpressionLists(String text) {
        Map<String, List<String>> out = new LinkedHashMap<>();
        Matcher matcher = Pattern.compile("\"([A-Za-z0-9_]+)\"\\s*:\\s*\\[(.*?)\\]", Pattern.DOTALL).matcher(text);
        while (matcher.find()) {
            String key = matcher.group(1);
            String body = matcher.group(2);
            List<String> values = new ArrayList<>();
            Matcher strings = Pattern.compile("\"(.*?)\"").matcher(body);
            while (strings.find()) {
                values.add(strings.group(1));
            }
            if (!values.isEmpty()) {
                out.put(key, values);
            }
        }
        return out;
    }

    private static double jsonDouble(String text, String key, double fallback) {
        Matcher matcher = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*([-+0-9.eE]+)").matcher(text);
        return matcher.find() ? Double.parseDouble(matcher.group(1)) : fallback;
    }

    private static boolean jsonBoolean(String text, String key, boolean fallback) {
        Matcher matcher = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*(true|false)").matcher(text);
        return matcher.find() ? Boolean.parseBoolean(matcher.group(1)) : fallback;
    }

    private static String jsonString(String text, String key, String fallback) {
        Matcher matcher = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*\"(.*?)\"").matcher(text);
        return matcher.find() ? matcher.group(1) : fallback;
    }

    private static String[] jsonStringArray(String text, String key, String[] fallback) {
        Matcher matcher = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*\\[(.*?)\\]", Pattern.DOTALL).matcher(text);
        if (!matcher.find()) {
            return fallback;
        }
        List<String> values = new ArrayList<>();
        Matcher strings = Pattern.compile("\"(.*?)\"").matcher(matcher.group(1));
        while (strings.find()) {
            values.add(strings.group(1));
        }
        return values.isEmpty() ? fallback : values.toArray(new String[0]);
    }

    private static String[] defaultAxisNames(int dim) {
        if (dim == 1) {
            return new String[]{"x"};
        }
        if (dim == 2) {
            return new String[]{"x", "y"};
        }
        return new String[]{"x", "y", "z"};
    }

    private static String firstNonEmpty(String... values) {
        for (String value : values) {
            if (value != null && !value.trim().isEmpty()) {
                return value;
            }
        }
        return "";
    }

    private static String sanitize(String value) {
        return value.replaceAll("[^A-Za-z0-9_]", "_");
    }

    private static String json(String value) {
        if (value == null) {
            return "null";
        }
        return "\"" + value.replace("\\", "\\\\").replace("\"", "\\\"").replace("\r", "\\r").replace("\n", "\\n") + "\"";
    }

    private static String jsonNumber(double value) {
        return Double.isFinite(value) ? String.format(Locale.US, "%.17g", value) : "null";
    }

    private static String jsonArray(String[] values) {
        List<String> quoted = new ArrayList<>();
        for (String value : values) {
            quoted.add(json(value));
        }
        return "[" + String.join(", ", quoted) + "]";
    }

    private static String jsonIntArray(int[] values) {
        List<String> formatted = new ArrayList<>();
        for (int value : values) {
            formatted.add(String.valueOf(value));
        }
        return "[" + String.join(", ", formatted) + "]";
    }

    private static String sha256(Path path) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        byte[] bytes = Files.readAllBytes(path);
        byte[] hash = digest.digest(bytes);
        StringBuilder sb = new StringBuilder();
        for (byte b : hash) {
            sb.append(String.format("%02x", b));
        }
        return sb.toString();
    }
}
