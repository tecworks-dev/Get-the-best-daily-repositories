package app.termora

import com.formdev.flatlaf.FlatClientProperties
import com.jetbrains.JBR
import com.jetbrains.WindowDecorations.CustomTitleBar
import java.awt.Rectangle
import java.awt.Window
import javax.swing.RootPaneContainer

class LogicCustomTitleBar(private val titleBar: CustomTitleBar) : CustomTitleBar {
    companion object {
        fun createCustomTitleBar(rootPaneContainer: RootPaneContainer): CustomTitleBar {
            if (!JBR.isWindowDecorationsSupported()) {
                return LogicCustomTitleBar(object : CustomTitleBar {
                    override fun getHeight(): Float {
                        val bounds = rootPaneContainer.rootPane
                            .getClientProperty(FlatClientProperties.FULL_WINDOW_CONTENT_BUTTONS_BOUNDS)
                        if (bounds is Rectangle) {
                            return bounds.height.toFloat()
                        }
                        return 0f
                    }

                    override fun setHeight(height: Float) {
                        rootPaneContainer.rootPane.putClientProperty(
                            FlatClientProperties.TITLE_BAR_HEIGHT,
                            height.toInt()
                        )
                    }

                    override fun getProperties(): MutableMap<String, Any> {
                        return mutableMapOf()
                    }

                    override fun putProperties(m: MutableMap<String, *>?) {

                    }

                    override fun putProperty(key: String?, value: Any?) {
                        if (key == "controls.visible" && value is Boolean) {
                            rootPaneContainer.rootPane.putClientProperty(
                                FlatClientProperties.TITLE_BAR_SHOW_CLOSE,
                                value
                            )
                        }
                    }

                    override fun getLeftInset(): Float {
                        return 0f
                    }

                    override fun getRightInset(): Float {
                        val bounds = rootPaneContainer.rootPane
                            .getClientProperty(FlatClientProperties.FULL_WINDOW_CONTENT_BUTTONS_BOUNDS)
                        if (bounds is Rectangle) {
                            return bounds.width.toFloat()
                        }
                        return 0f
                    }

                    override fun forceHitTest(client: Boolean) {

                    }

                    override fun getContainingWindow(): Window {
                        return rootPaneContainer as Window
                    }
                })
            }
            return JBR.getWindowDecorations().createCustomTitleBar()
        }
    }

    override fun getHeight(): Float {
        return titleBar.height
    }

    override fun setHeight(height: Float) {
        titleBar.height = height
    }

    override fun getProperties(): MutableMap<String, Any> {
        return titleBar.properties
    }

    override fun putProperties(m: MutableMap<String, *>?) {
        titleBar.putProperties(m)
    }

    override fun putProperty(key: String?, value: Any?) {
        titleBar.putProperty(key, value)
    }

    override fun getLeftInset(): Float {
        return titleBar.leftInset
    }

    override fun getRightInset(): Float {
        return titleBar.rightInset
    }

    override fun forceHitTest(client: Boolean) {
        titleBar.forceHitTest(client)
    }

    override fun getContainingWindow(): Window {
        return titleBar.containingWindow
    }
}