#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
APP_NAME="thinkfarm-client.app"
APP_PATH="${SCRIPT_DIR}/dist/${APP_NAME}"
DMG_NAME="thinkfarm-client-macOS.dmg"
DMG_PATH="${SCRIPT_DIR}/dist/${DMG_NAME}"
STAGING_DIR="${SCRIPT_DIR}/dist/dmg_staging"
IDENTITY="Developer ID Application: paul xavier perrine (B4LQ2JCC67)"

echo "=== Creating DMG Packaging for ${APP_NAME} ==="

if [ ! -d "${APP_PATH}" ]; then
    echo "Error: ${APP_PATH} does not exist. Run PyInstaller and signing first."
    exit 1
fi

# Prepare staging directory
rm -rf "${STAGING_DIR}" "${DMG_PATH}"
mkdir -p "${STAGING_DIR}"

echo "Copying app to staging..."
cp -R "${APP_PATH}" "${STAGING_DIR}/"

echo "Creating Applications folder shortcut..."
ln -s /Applications "${STAGING_DIR}/Applications"

echo "Building DMG using hdiutil..."
hdiutil create -volname "thinkfarm-client" -srcfolder "${STAGING_DIR}" -ov -format UDZO "${DMG_PATH}"

echo "Cleaning up staging directory..."
rm -rf "${STAGING_DIR}"

echo "Signing DMG with Developer ID..."
codesign --force --sign "${IDENTITY}" "${DMG_PATH}"

echo "Verifying DMG signature..."
codesign -vvv --strict "${DMG_PATH}"

echo ""
echo "=== DMG Created Successfully! ==="
echo "Location: ${DMG_PATH}"
echo ""
echo "Next Steps for Distribution:"
echo "1. Notarize the DMG file:"
echo "   xcrun notarytool submit ${DMG_PATH} --keychain-profile \"AC_PASSWORD\" --wait"
echo ""
echo "2. Staple the notarization ticket to the DMG:"
echo "   xcrun stapler staple ${DMG_PATH}"
echo ""
echo "3. Verify Gatekeeper acceptance:"
echo "   spctl --assess --type execute --verbose ${APP_PATH}"
